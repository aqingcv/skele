python main.py pretrain --config \
./config/pretrain/pretrain_skeletonsim.yaml
tensorboard --logdir=/data1/zengyq/skeletonsim/work_dir/ntu60xview/model

python main.py linear_evaluation \
--config ./config/linear_eval/linear_eval_skeletonsim.yaml \
--weights /data1/zengyq/skeletonsim/work_dir/ntu60xview/modellr01/epoch300_model.pt

python main.py finetune_evaluation \
--config config/fine_tune/fine_tune_skeletonbyol.yaml \
--weights /data1/zengyq/skeletonsim/work_dir/ntu60xview/modelsmall/epoch10_model.pt


lr 0.05 68.12

lr 0.02 60.75
