#!/usr/bin/env bash
# Knockoff Filter Experiment (Sections 4.1 and 4.2)
#
# Each method section runs independently and skips missing files:
#
#   L2D knockoff      -- requires .l2d.json in exp_diverse/results/
#   ImBD knockoff     -- requires .imbd.json in exp_diverse/results/
#   Likelihood knockoff -- requires:
#                           exp_diverse/data/*.raw_data.json  (raw texts)
#                           exp_diverse/results/*.rewrite_4.json  (LLM rewrites)
#                         These are produced by detect_l2d.py but can also
#                         exist independently (e.g. pre-downloaded).
#
# Outputs (per dataset × source model):
#   .knockoff_l2d.json        -- L2D knockoff FDR/power table
#   .knockoff_imbd.json       -- ImBD knockoff FDR/power table
#   .likelihood_knockoff.json -- likelihood signed stats (g(R)-g(T))
#   .knockoff_likelihood.json -- likelihood knockoff FDR/power table

set -e
echo "$(date)  Starting knockoff experiment ..."

exp_path=exp_diverse
data_path=$exp_path/data
res_path=$exp_path/results
mkdir -p $res_path

source_models="Llama-3-70B GPT-3-Turbo Gemini-1.5-Pro GPT-4o"
datasets="AcademicResearch EducationMaterial FoodCusine MedicalText ProductReview \
TravelTourism ArtCulture Entertainment GovernmentPublic NewsArticle Religious \
Business Environmental LegalDocument OnlineContent Sports Finance \
PersonalCommunication TechnicalWriting"

scoring_model="gemma-1b"
q_levels="0.05 0.1 0.2 0.3 0.5"
gpu_device="cuda"
cache_dir=".cache/huggingface"

# ── L2D knockoff ─────────────────────────────────────────────────────────────
echo "$(date)  Applying knockoff filter to L2D results ..."
for D in $datasets; do
  for M in $source_models; do
    l2d_file="$res_path/${D}_${M}.l2d.json"
    if [ -f "$l2d_file" ]; then
      python scripts/detect_knockoff.py \
        --results_file  "$l2d_file" \
        --method        l2d \
        --output_file   "$res_path/${D}_${M}.knockoff_l2d.json" \
        --q_levels      $q_levels
    else
      echo "  [skip] $l2d_file not found"
    fi
  done
done

# ── ImBD knockoff ─────────────────────────────────────────────────────────────
echo "$(date)  Applying knockoff filter to ImBD results ..."
for D in $datasets; do
  for M in $source_models; do
    imbd_file="$res_path/${D}_${M}.imbd.json"
    if [ -f "$imbd_file" ]; then
      python scripts/detect_knockoff.py \
        --results_file  "$imbd_file" \
        --method        imbd \
        --output_file   "$res_path/${D}_${M}.knockoff_imbd.json" \
        --q_levels      $q_levels
    else
      echo "  [skip] $imbd_file not found"
    fi
  done
done

# ── Likelihood knockoff (requires rewrite data from L2D pipeline) ─────────────
echo "$(date)  Computing likelihood knockoff statistics ..."
for D in $datasets; do
  for M in $source_models; do
    data_file="$data_path/${D}_${M}"
    rewrite_file="$res_path/${D}_${M}.rewrite_4.json"
    if [ -f "${data_file}.raw_data.json" ] && [ -f "$rewrite_file" ]; then
      # Step 1: compute signed stats g(R_i) - g(T_i)
      python scripts/detect_likelihood.py \
        --dataset_file        "$data_file" \
        --scoring_model_name  "$scoring_model" \
        --output_file         "$res_path/${D}_${M}" \
        --rewrite_file        "$rewrite_file" \
        --device              "$gpu_device" \
        --cache_dir           "$cache_dir" \
        --knockoff

      # Step 2: apply knockoff filter to the signed stats
      python scripts/detect_knockoff.py \
        --results_file  "$res_path/${D}_${M}.likelihood_knockoff.json" \
        --method        likelihood \
        --output_file   "$res_path/${D}_${M}.knockoff_likelihood.json" \
        --q_levels      $q_levels
    else
      echo "  [skip] data or rewrite not found for ${D}_${M}"
    fi
  done
done

echo "$(date)  Knockoff experiment complete. Results in $res_path/"
