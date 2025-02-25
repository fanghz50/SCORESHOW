# 

## 环境配置

```shell
conda create -n scoreshow python=3.11.11
conda activate scoreshow
#自己的环境
conda list > conda_list.txt
#导出为requirements_pip.txt和requirements_conda.txt
python to_requirements.py
#安装环境
pip install -r requirements_pip.txt
conda install --file requirements_conda.txt -c conda-forge
```

## pytorch我是用的CPU版本，想用gpu需要找匹配版本

```shell
Windows: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
Linux: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Change other pytorch version in  [![Pytorch](https://img.shields.io/badge/PYtorch-test?style=flat&logo=pytorch&logoColor=white&color=orange)](https://pytorch.org/)





# 4. 添加字体

## Windows User

Copy all font files `*.ttf` in `fonts` folder into `C:\Windows\Fonts`

## Linux User

```shell
mkdir -p ~/.local/share/fonts
sudo cp fonts/Shojumaru-Regular.ttf ~/.local/share/fonts/
sudo fc-cache -fv
```

# 5. Run Program

```shell
python main.py
```

## Frames

[![Python](https://img.shields.io/badge/python-3776ab?style=for-the-badge&logo=python&logoColor=ffd343)](https://www.python.org/)[![Pytorch](https://img.shields.io/badge/PYtorch-test?style=for-the-badge&logo=pytorch&logoColor=white&color=orange)](https://pytorch.org/)[![Static Badge](https://img.shields.io/badge/Pyside6-test?style=for-the-badge&logo=qt&logoColor=white)](https://doc.qt.io/qtforpython-6/PySide6/QtWidgets/index.html)

## Reference

