import json
import torch

import copy
import warnings
from datetime import timedelta
from typing import List, Optional, Tuple, Union

from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from packaging import version
from PIL import Image
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.utils import stop_sequences_criteria

warnings.filterwarnings("ignore")

from loguru import logger as eval_logger

try:
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
    from llava.conversation import conv_templates, default_conversation
except Exception as e:
    eval_logger.debug("LLaVA is not installed. Please install LLaVA to use this model.\nError: %s" % e)


@register_model("babyllava")
class BabyLlava(lmms):
    """
    BabyLlava Model
    Supports single-turn (generate_until) and multi-turn (generate_until_multi_round) inference.
    Multi-turn is used for the memory task which requires iterative conversation with accumulated images.
    """
    def __init__(
        self,
        pretrained: str = "/projectnb/ivc-ml/wsashawn/LLaVA/checkpoints/babyllava_default",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        model_name=None,
        device_map="cuda:0",
        conv_template="baby_v1",
        use_cache=True,
        tie_weights: bool = True,
        truncate_context=False,
        customized_config=None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        disable_torch_init()
        model_path = pretrained
        model_name = model_name if model_name is not None else get_model_name_from_path(model_path)
        model_base = "liuhaotian/llava-v1.5-7b" if 'lora' in model_name else None
        self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(
            model_path, model_base, model_name,
            device_map=self.device_map,
            use_flash_attn=True,
        )
        self._config = self._model.config
        self.model.eval()
        if tie_weights:
            self.model.tie_weights()
        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")
            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._world_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            image_sizes = [[visual.size[0], visual.size[1]] for visual in visuals]
            if visuals:
                image = process_images(visuals, self._image_processor, self._config)
                if type(image) is list:
                    image = [_image.to(dtype=torch.float16, device=self.device) for _image in image]
                else:
                    image = image.to(dtype=torch.float16, device=self.device)
            else:
                image = None

            prompts_input = contexts[0] if isinstance(contexts, list) else contexts

            if image is not None and len(image) != 0 and DEFAULT_IMAGE_TOKEN not in prompts_input:
                image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visuals)
                image_tokens = " ".join(image_tokens)
                prompts_input = image_tokens + "\n" + (contexts[0] if isinstance(contexts, list) else contexts)

            if "llama_3" in self.conv_template:
                conv = copy.deepcopy(conv_templates[self.conv_template])
            else:
                conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], prompts_input)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            contxt_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            conv.messages[1][1] = continuation

            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            labels = input_ids.clone()
            labels[0, : contxt_id.shape[1]] = -100
            with torch.inference_mode():
                outputs = self.model(input_ids=input_ids, labels=labels, images=image, use_cache=True, image_sizes=image_sizes)
            loss = outputs["loss"]
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = input_ids[:, contxt_id.shape[1]:]
            greedy_tokens = greedy_tokens[:, contxt_id.shape[1]: input_ids.shape[1]]
            max_equal = (greedy_tokens == cont_toks).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)
        pbar.close()
        return res

    def flatten(self, input):
        if not input or any(i is None for i in input):
            return []
        new_list = []
        for i in input:
            if i:
                for j in i:
                    new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for req in requests:
            ctx, gen_kwargs, doc_to_visual, doc_id, task, split = req.args
            doc = self.task_dict[task][split][doc_id]
            visual = doc_to_visual(doc)
            visuals = self.flatten([visual])

            gen_kwargs = dict(gen_kwargs)
            until = [self.tok_decode(self.eot_token_id)]

            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")

            if "image_aspect_ratio" in gen_kwargs.keys() and "image_aspect_ratio" not in self._config.__dict__:
                self._config.image_aspect_ratio = gen_kwargs.pop("image_aspect_ratio")
                eval_logger.info(f"Setting image aspect ratio: {self._config.image_aspect_ratio}")

            if visuals:
                image_tensor = process_images(visuals, self._image_processor, self._config)
                if type(image_tensor) is list:
                    image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                else:
                    image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
            else:
                image_tensor = None

            if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in ctx:
                image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visual) if isinstance(visual, list) else [DEFAULT_IMAGE_TOKEN]
                question = " ".join(image_tokens) + "\n" + ctx
            else:
                question = ctx

            if "llama_3" in self.conv_template:
                conv = copy.deepcopy(conv_templates[self.conv_template])
            else:
                conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            gen_kwargs.setdefault("max_new_tokens", 1024)
            gen_kwargs.setdefault("temperature", 0)
            gen_kwargs.setdefault("top_p", None)
            gen_kwargs.setdefault("num_beams", 1)
            gen_kwargs["image_sizes"] = [v.size for v in visuals]

            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            attention_masks = input_ids.ne(pad_token_ids).to(self.device)

            try:
                output_ids = self.model.generate(
                    input_ids,
                    attention_mask=attention_masks,
                    pad_token_id=pad_token_ids,
                    images=image_tensor,
                    image_sizes=gen_kwargs["image_sizes"],
                    do_sample=True if gen_kwargs["temperature"] > 0 else False,
                    temperature=gen_kwargs["temperature"],
                    top_p=gen_kwargs["top_p"],
                    num_beams=gen_kwargs["num_beams"],
                    max_new_tokens=gen_kwargs["max_new_tokens"],
                    use_cache=self.use_cache,
                )
                output = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                for u in until:
                    output = output.split(u)[0]
            except Exception as e:
                eval_logger.error(f"Error {e} in generating for task={task}, split={split}, doc_id={doc_id}")
                output = ""

            res.append(output)
            self.cache_hook.add_partial("generate_until", (ctx, gen_kwargs), [output])
            pbar.update(1)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        """
        Multi-turn generation for tasks like memory where images and conversation
        are accumulated across multiple rounds.

        Each request's args: (ctx, gen_kwargs, doc_to_visual, doc_to_text_fn, doc_id, task, split)

        Returns a JSON-encoded list of per-round responses for each request.
        The task's process_results function decodes this JSON to compute metrics.
        """
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding (multi-round)")

        for req in requests:
            ctx, gen_kwargs, doc_to_visual, doc_to_text_fn, doc_id, task, split = req.args
            doc = self.task_dict[task][split][doc_id]

            all_image_paths = doc["image"]
            conversations = doc["conversations"]

            # Build conversation iteratively, mirroring test_baby.py's test_single_sample logic
            conv = conv_templates[self.conv_template].copy()
            role_user, role_bot = conv.roles

            all_images = []      # accumulated image tensors across all rounds
            all_sizes = []       # accumulated image sizes
            all_images_count = 0
            responses = []

            max_new_tokens = gen_kwargs.get("max_new_tokens", 128)
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id

            for round_idx, chat_round in enumerate(conversations):
                if chat_round["from"] != "human":
                    continue

                # Count <image> tokens in this human turn
                image_count = chat_round["value"].count(DEFAULT_IMAGE_TOKEN)

                # Load and process images for this round
                round_image_paths = all_image_paths[all_images_count: all_images_count + image_count]
                all_images_count += image_count

                if round_image_paths:
                    imgs = [Image.open(p).convert("RGB") for p in round_image_paths]
                    tensors = process_images(imgs, self._image_processor, self._config)
                    if isinstance(tensors, list):
                        tensors = [t.to(dtype=torch.float16, device=self.device) for t in tensors]
                    else:
                        tensors = [tensors.to(dtype=torch.float16, device=self.device)]
                    all_images.extend(tensors)
                    all_sizes.extend([img.size for img in imgs])

                # Add user message to running conversation
                conv.append_message(role_user, chat_round["value"])

                # If next round is also from human, accumulate without generating
                next_idx = round_idx + 1
                if next_idx < len(conversations) and conversations[next_idx]["from"] == "human":
                    continue

                # Generate response for this round
                conv.append_message(role_bot, None)
                prompt = conv.get_prompt()

                input_ids = tokenizer_image_token(
                    prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                ).unsqueeze(0).to(self.device)
                attention_mask = input_ids.ne(pad_token_id).to(self.device)

                with torch.inference_mode():
                    output_ids = self.model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        pad_token_id=pad_token_id,
                        images=all_images,
                        image_sizes=all_sizes,
                        do_sample=False,
                        max_new_tokens=max_new_tokens,
                        use_cache=self.use_cache,
                    )

                response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

                # Store response in conversation history for subsequent rounds
                conv.messages[-1][1] = response
                responses.append(response)

            res.append(json.dumps(responses))
            pbar.update(1)

        pbar.close()
        return res
