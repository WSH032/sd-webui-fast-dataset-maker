$tags = "hifumi_(blue_archive)"    # 符合gelbooru搜索规则的tags | tags for gelbooru
$max_images_number = 200    # 需要下载的图片数 | the number of images you need to download
$download_dir = "images"    # 下载图片的路径 | the folder path to download images
$max_workers= 15    # 最大下载协程数 | maximum number of download coroutines
$unit = 100    # 下载单位，下载图片数以此向上取一单位 | unit for download. eg: max_images_number=11, unit=10, then you get 20
$timeout = 10    # 下载超时限制 | download connecting timeout limit


##########  你可以在这找到tags规则 some useful info for tags   ##########
<# 

  API  https://gelbooru.com/index.php?page=wiki&s=view&id=18780
  tags  https://gelbooru.com/index.php?page=wiki&s=&s=view&id=25921
  cheatsheet  https://gelbooru.com/index.php?page=wiki&s=&s=view&id=26263

#>
##########  下载脚本 do not edit  ##########
.\venv\Scripts\activate

python download_images_coroutine.py `
  --tags=$tags `
  --max_images_number=$max_images_number `
  --download_dir=$download_dir `
  --max_workers=$max_workers `
  --unit=$unit `
  --timeout=$timeout
  
pause