# 贡献工作流

## Bump 扩展版本

### 1.基础检查
- 检查子扩展的`requirements.txt`或者`install.py`是否有更新
  - 如果有，请修改本项目的`requirements.txt`文件
- 检查子扩展的`UI`组件部分是否更新
  - 如果有，请修改`extensions/extensions_ui.py`
- 检查子扩展的`preload`部分或者`preload.py`是否更新
  - 如果有，请修改`extensions/extensions_preload.py`
- 检查子扩展的`style.css`文件是否改名或移动位置；检查`js`文件是否仍在原先的`javascript`文件夹下；特别是`dataset-tag-editor-standalone`扩展
  - 如果发生改变，请修改`extensions/extensions_ui.py`中的`css_str`和`js_str`
- 运行并观察控制台是否有警告和报错

### 2.高级检查
- 检查子扩展的`JavaScript`部分是否更新，特别是`sd-webui-infinite-image-browsing`扩展
  - 打开浏览器控制台，测试`js`功能，观察是否有报错
    - 如果有，一个可能原因是，子扩展使用了`A1111-WebUI`自带的`js函数`，而本项目没有提供
    - 请参考[AUTOMATIC1111/stable-diffusion-webui/blob/master/script.js](https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/script.js)和[AUTOMATIC1111/stable-diffusion-webui/tree/master/javascript](https://github.com/AUTOMATIC1111/stable-diffusion-webui/tree/master/javascript)对`script.js`进行修改

### 3.Colab测试
- 在Colab中更新(如有必要)`fast_dataset_maker.ipynb`并进行测试

### 4.修改`README.md`

### 5.发起`PR`
