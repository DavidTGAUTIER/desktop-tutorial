For local use (on localhost:4000): 
docker build . -t getaround
windows : docker run -it -v "%cd%:/home/app" -p 4000:4000 -e PORT=4000 getaround
linux   : docker run -it -v "$(pwd):/home/app" -p 4000:4000 -e PORT=4000 getaround