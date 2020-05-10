#!/bin/bash

MINPARAMS=1
if [ $# -ne "$MINPARAMS" ]
then
  echo
  echo "This script needs $MINPARAMS command-line argument(s)!"
  exit 0
fi 

for ARGUMENT in "$@"
do
    key=$(echo $ARGUMENT | cut -f1 -d=)
    value=$(echo $ARGUMENT | cut -f2 -d=)   

    case "$key" in
            step)               step=${value} ;;
            *)   
    esac    


done

is_valid_opt=false

for opt in COLORIZED_DEPTH_SAVE FIX_EXTRACTION FIX_RECURSIVE_NN FINE_TUNING FINE_EXTRACTION FINE_RECURSIVE_NN
do
  if [ "$step" = "$opt" ]
  then
    is_valid_opt=true
  fi
done

if [ "$is_valid_opt" = false ]
then
  echo
  echo 'Invalid option: "'$step'"! Should be one of the below'
  echo 'COLORIZED_DEPTH_SAVE, FIX_EXTRACTION, FIX_RECURSIVE_NN, FINE_TUNING, FINE_EXTRACTION, FINE_RECURSIVE_NN'
  exit 0
fi

run_step='    proceed_step = RunSteps.'$step
line_num="$(grep -n "    proceed_step = RunSteps." main_steps.py | tail -n 1 | cut -d: -f1)"

# below code line replace the related line number in main_steps.py.
sed -i $line_num's/.*/'"$run_step"'/' main_steps.py

