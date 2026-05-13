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

scoring_model="gemma-9b-instruct"
rewrite_model="gemma-9b-instruct"
likelihood_scoring_model="gemma-1b"
q_levels="0.05 0.1 0.2 0.3 0.5"
gpu_device="cuda"
cache_dir=".cache/huggingface"

# ── ImBD knockoff (signed: W_i = f(T_i) - f(R_i)) ───────────────────────────
echo "$(date)  Computing IMBD knockoff signed statistics ..."
trained_imbd_path=scripts/ImBD/ckpt/ai_detection_500_spo_lr_0.0001_beta_0.05_a_1
for D in $datasets; do
  for M in $source_models; do
    imbd_file="$res_path/${D}_${M}.imbd.json"
    data_file="$data_path/${D}_${M}"
    rewrite_file="$res_path/${D}_${M}.rewrite_4.json"
    if [ -f "$imbd_file" ] && [ -f "${data_file}.raw_data.json" ] && [ -f "$rewrite_file" ]; then
      # Step 1: compute signed stats f(T_i) - f(R_i)
      echo ""
    else
      echo "  [skip] missing files for ${D}_${M}"
    fi
  done
done

