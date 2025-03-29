# Use a Windows base image
FROM ubuntu:20.04 

# Set environment variables
ENV PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Run as root user
USER root

# Update package list and install necessary packages
RUN apt-get update && \
    apt-get install -y curl git apt-transport-https ca-certificates gnupg

# Install Go
RUN curl -LO https://golang.org/dl/go1.17.2.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go1.17.2.linux-amd64.tar.gz && \
    rm go1.17.2.linux-amd64.tar.gz

ENV PATH="/usr/local/go/bin:${PATH}"

# Install Hugo
RUN curl -L -o hugo.tar.gz https://github.com/gohugoio/hugo/releases/download/v0.115.4/hugo_extended_0.115.4_Linux-64bit.tar.gz && \
    tar -C /usr/local/bin -xzf hugo.tar.gz && \
    rm hugo.tar.gz

# Install Google Cloud SDK
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | \
    tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
    gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    apt-get update -y && \
    apt-get install google-cloud-cli -y

# Clean up
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /site

# 
COPY blog /site/

EXPOSE 1313

# Default command
CMD ["bash"]