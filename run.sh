# Prompt is not used for video2world, but we keep it here for consistency (Will be removed in the future)
PROMPT=""

python -m examples.video2world \
    --model_size 2B \
    --dit_path "checkpoints/Re2Pix_cs_40k/re2pix_cs_40k.pt" \
    --resolution 480 \
    --input_path datasets/benchmark_train/cityscapes_seq_video/val/videos/cs_sec_0.mp4 \
    --fps 16 \
    --num_conditional_frames 13 \
    --prompt "${PROMPT}" \
    --disable_guardrail \
    --disable_prompt_refiner \
    --guidance 0.0 \
    --num_comp 1152 \
    --save_path output/cs_sec_0_re2pix.mp4