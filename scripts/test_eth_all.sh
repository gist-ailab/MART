#!/bin/bash

# Handle SIGINT (CTRL+C) to kill all child processes
trap "kill 0" SIGINT

# Run processes in parallel
python main_eth.py --config "$1" --gpu "$2" --dataset eth --test &
python main_eth.py --config "$1" --gpu "$2" --dataset hotel --test &
python main_eth.py --config "$1" --gpu "$2" --dataset univ --test &
python main_eth.py --config "$1" --gpu "$2" --dataset zara1 --test &
python main_eth.py --config "$1" --gpu "$2" --dataset zara2 --test &
wait

# List of result files
result_files=(
  "./results/eth_result.csv"
  "./results/hotel_result.csv"
  "./results/univ_result.csv"
  "./results/zara1_result.csv"
  "./results/zara2_result.csv"
)

# Initialize sums and arrays for FDE and ADE
minfde_values=()
minade_values=()
minfde_sum=0
minade_sum=0
count=0

# Dataset names
datasets=("ETH" "HOTEL" "UNIV" "ZARA1" "ZARA2")

# Loop through each result file
for file in "${result_files[@]}"; do
  if [ -f "$file" ]; then
    # Read the first non-blank line
    first_line=$(head -n 1 "$file" | tr -d '\r')  # Remove potential ^M characters
    if [ -n "$first_line" ]; then
      # Extract fde and ade values (adjusted order: configname,fde,ade)
      IFS=',' read -r config minfde minade <<< "$first_line"
      if [[ $minfde =~ ^[0-9]+(\.[0-9]+)?$ && $minade =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        minfde_values+=("$(printf "%.2f" "$minfde")")
        minade_values+=("$(printf "%.2f" "$minade")")
        minfde_sum=$(echo "$minfde_sum + $minfde" | bc)
        minade_sum=$(echo "$minade_sum + $minade" | bc)
        count=$((count + 1))
      else
        echo "Skipping invalid line in $file: $first_line"
        minfde_values+=("N/A")
        minade_values+=("N/A")
      fi
    else
      minfde_values+=("N/A")
      minade_values+=("N/A")
    fi
  else
    echo "Warning: $file not found."
    minfde_values+=("N/A")
    minade_values+=("N/A")
  fi
done

# Calculate averages
if [ "$count" -gt 0 ]; then
  minfde_avg=$(printf "%.2f" $(echo "scale=2; $minfde_sum / $count" | bc))
  minade_avg=$(printf "%.2f" $(echo "scale=2; $minade_sum / $count" | bc))
else
  minfde_avg="N/A"
  minade_avg="N/A"
fi

# Summarize results in table format
echo ""
echo "minADE Table"
echo -n "       "
for dataset in "${datasets[@]}"; do
  echo -n "$dataset    "
done
echo "AVG"
echo -n "       "
for ade in "${minade_values[@]}"; do
  printf "%-8s" "$ade"
done
printf "%-8s\n" "$minade_avg"

echo ""
echo "minFDE Table"
echo -n "       "
for dataset in "${datasets[@]}"; do
  echo -n "$dataset    "
done
echo "AVG"
echo -n "       "
for fde in "${minfde_values[@]}"; do
  printf "%-8s" "$fde"
done
printf "%-8s\n" "$minfde_avg"
