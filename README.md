# 图书推荐系统

## 项目简介

这是一个图书推荐系统，使用的数据集为高校图书馆近几年的借阅数据，通过使用DSSM深度语义匹配模型对其中的用户进行个性化推荐，本仓库是项目的后端实现，具体的前端界面可访问[图书管理系统](https://mikus.love/book/login)，本项目旨在通过个性化推荐系统实现高校图书馆资源的充分利用。

项目的github地址：[BookRecSystem](https://github.com/Mikeaser/BookRecSystem)

## 运行说明

教程位于项目目录下的文件BookRecTutorial.ipynb当中

环境要求位于项目目录下的目录Requirements中

包含windows与linux系统的环境信息

想要复刻环境，终端运行

conda install --torchrec --file ./requirements/requirements_linux64.txt

conda install --torchrec --file ./requirements/requirements_win64.txt

第一个参数为环境名字，可以任取，第二个参数后面是文件的路径，选择对应的系统安装就行

**注意**

annoy包需要系统带有C++编译环境，终端执行pip install annoy

实验所用的torch版本大于等于1.11.0均可，cuda大于等于11.3，更低的版本未经测试。

## 目录结构

项目的目录结构大致如图所示：
