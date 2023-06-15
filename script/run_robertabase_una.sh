python run.py \
    --epochs 1 \
    --batch_size 64 \
    --batch_size_eval 256 \
    --lr 5e-6 \
    --max_len 32 \
    --pooling cls \
    --activation tanh \
    --temperature 0.05 \
    --dropout 0.1 \
    --pretrained_model roberta-base \
    --num_workers 0 \
    --do_para \
    --aug_every_n_steps 5 \
    --do_hard_negatives \
    --do_tf_idf \
    --token_prob 0.5 \
    --tfidf_mode nbh \
    --seed 42 \
    --weight_decay 0.0 \
    --log_every_n_steps 100 \
    --fp16_precision \
    --train_data ./data/training/wiki1m_for_simcse.txt \
    --para_train_data ./data/training/paraphrase.txt \
    --dev_file ./data/stsbenchmark/sts-dev.csv \
    --test_file data/stsbenchmark/sts-test.csv \
    --save_data \
    --output_path roberta_una \