# electric-theft-detection

# preprocessing 

Check out the preprocessing steps at
`/notebooks/visualization.ipynb`

# training

```
CUDA_VISIBLE_DEVICES=2 python3 trainer.py
```

with the default params:
```
    parser.add_argument('--indir', type=str, default="data/ksm_transformer_best_result")
    parser.add_argument('--model', type=str, default="old")
    parser.add_argument('--log_mode', type=str, default="train")
    parser.add_argument('--window_size', type=int, default=37)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--mode', type=str, default='transformer')
    parser.add_argument('--model_name', type=str, default='KSM_Transformer')
    parser.add_argument('--model_path', type=str, default='model/')
```