cd ~
mkdir work_space Git data
mkdir work_space/log work_space/history work_space/models work_space/files work_space/save
mkdir data/ISIC

cd Git
git clone https://github.com/GNovich/CKA.git
ln -sfn ~/data data

cd CKA/work_space
ln -sfn ~/work_space/log log
ln -sfn ~/work_space/history history
ln -sfn ~/work_space/save save
ln -sfn ~/work_space/models models
ln -sfn ~/work_space/files files

# download dataset
cd ~/data
cp -dr /mnt/md0/orville/Gal/ISIC.zip .
uzip ISIC.zip
rm ISIC.zip
cd ~

conda env create -f Git/CKA/environment.yml

