# HMDRN

Code release for [Hierarchical Mask-Enhanced Dual Reconstruction Network for Few-Shot Fine-Grained Image Classification]

## Code environment

* You can create a conda environment with the correct dependencies using the following command lines:

  ```shell
  conda env create -f environment.yaml
  conda activate HMDRN
  ```

## Dataset

The official link of CUB-200-2011 is [here](http://www.vision.caltech.edu/datasets/cub_200_2011/). The preprocessing of the cropped CUB-200-2011 is the same as [FRN](https://github.com/Tsingularity/FRN), but the categories of train, val, and test follows split.txt. And then move the processed dataset to directory ./data.

- CUB_200_2011 \[[Download Link](https://drive.google.com/file/d/1WxDB3g3U_SrF2sv-DmFYl8LS0p_wAowh/view)\]
- cars \[[Download Link](https://drive.google.com/file/d/1ImEPQH5gHpSE_Mlq8bRvxxcUXOwdHIeF/view?usp=drive_link)\]
- dogs \[[Download Link](https://drive.google.com/file/d/13avzK22oatJmtuyK0LlShWli00NsF6N0/view?usp=drive_link)\]

## Train

* To train HMDRN on `CUB_fewshot_cropped` with Conv-4 backbone under the 1/5-shot setting, run the following command lines:

  ```shell
  cd experiments/CUB_fewshot_cropped/HMDRN/Conv-4
  ./train.sh
  ```
  
* For ResNet-12 backbone, run the following command lines:

  ```shell
  cd experiments/CUB_fewshot_cropped/HMDRN/ResNet-12
  ./train.sh
  ```

## Test

```shell
    cd experiments/CUB_fewshot_cropped/HMDRN/Conv-4
    python ./test.py
    
    cd experiments/CUB_fewshot_cropped/HMDRN/ResNet-12
    python ./test.py
```

## References

Thanks to [Davis](https://github.com/Tsingularity/FRN), [Phil](https://github.com/lucidrains/vit-pytorch) and [Yassine](https://github.com/yassouali/SCL), for the preliminary implementations.

## Contact

Thanks for your attention!
If you have any suggestion or question, you can leave a message here or contact us directly:

- ningluo@email.ncu.edu.cn
