# sd-fast-dataset-maker-webui
Scrape images from Gelbooru
Deduplicate and cluster by WebUI

## 现在我们有什么？
 - 一个访问Gelbooru API来下载图片的脚本
 - 基于imagededup库，进行图片去重的WebUI
 - 基于sklearn库，以tags为特征的图片聚类WebUI

## 使用方法
所有内容目前均以脚本形式提供
稍后进行封装

## Todo
- [ ] 在Colab上部署
- [ ] 完成本地部署封装
- [ ] 为下载图片添加WebUI
- [ ] 为图片聚类增加SVD降维，更多聚类方式与参数

## 部分展示
![deduplicate_demo](./docs/deduplicate_demo.png)

![cluster_demo_0](./docs/cluster_demo_0.png)

![cluster_demo_1](./docs/cluster_demo_1.png)
