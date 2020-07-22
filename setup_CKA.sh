cd ~
mkdir work_space Git data
mkdir work_space/log work_space/history work_space/models work_space/files work_space/save
mkdir data/ISIC

cd Git
git clone https://github.com/GNovich/CKA.git
ln -sfn ~/data CKA/data
ln -sfn ~/work_space CKA/work_space

# download dataset
cd ~/data
cp -dr /mnt/md0/orville/Gal/ISIC.zip .
unzip ISIC.zip
rm ISIC.zip
mkdir ISIC/Test
mv ISIC/ISIC_2019_Test_Input ISIC/Test/
cd ~

conda env create -f Git/CKA/environment.yml
