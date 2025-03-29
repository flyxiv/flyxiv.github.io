# 1. Create Image 

```sh
$ docker build -t blog_dev_env:v0.1 ./dockerfiles
```

# 2. Download and Autenticate gcloud
```sh
# Run gcloud init with the installer
$ (New-Object Net.WebClient).DownloadFile("https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe", "$env:Temp\GoogleCloudSDKInstaller.exe") & $env:Temp\GoogleCloudSDKInstaller.exe

$ gcloud auth login

# Copy Config settings in C:\Users\<user>\AppData\Roaming\gcloud:/root/.config/gcloud to ./gcloud

# Now move credentials and blog directory into the container
$ docker run -it --name blog-dev-container -v "$(pwd)/blog:/site/blog" -v "$(pwd)/gcloud:/root/.config/gcloud" blog_dev_env:v0.1 bash 
```

# 3. Run Container 
```sh
$ docker run -it --name blog-dev-container -v "$(pwd)/blog:/site/blog" -v "./gcloud:/root/.config/gcloud" -p 1313:1313 blog_dev_env:v0.1 bash  
```

# 4. Generate Hugo Static Files and upload to google cloud
```sh
# In container blog-dev-container
$ hugo


# Output ex)
#                   | EN
#-------------------+-----
#  Pages            |  7
#  Paginator pages  |  0
#  Non-page files   |  0
#  Static files     |  0
#  Processed images |  0
#  Aliases          |  3
#  Sitemaps         |  1
#  Cleaned          |  0

$ gsutil cp -r public gs://jyn_blog_static_files
```
