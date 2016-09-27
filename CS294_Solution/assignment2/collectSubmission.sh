rm -f assignment2.zip
zip -r assignment2.zip . -x "*.git*" "*cs294_129/datasets*" "*.ipynb_checkpoints*" "*README.md" "*collectSubmission.sh" "*requirements.txt" ".env/*" "*.pyc" "*cs294_129/build/*"
