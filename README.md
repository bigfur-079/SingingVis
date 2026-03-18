# SingingVis

### 建立虛擬環境
python版本要有3.11以上 <br>
```sudo apt update``` <br>
```sudo apt install python3.11```

建立虛擬機，並安裝 requirements.txt 的套件 <br>
```python3.11 -m venv venv``` <br>
```source venv/bin/activate``` <br>
```pip install requirements.txt``` <br>

### 設定 Flask
安裝 gunicorn <br>
```source venv/bin/activate``` <br>
```pip install gunicorn``` 

測試 gunicorn <br>
```gunicorn --bind 0.0.0.0:5000 /flask/singingvis_api_ubuntu:app```

建立 systemd 服務（達成持續執行）<br>
```sudo nano /etc/systemd/system/singingvis.service``` <br>
```
[Unit]
Description=Gunicorn instance to serve SingingVis
After=network.target

[Service]
# 你的 Ubuntu 使用者名稱
User=m134020017
Group=www-data
# 專案根目錄路徑
WorkingDirectory=/var/www/html/SingingVis
# 虛擬環境中 gunicorn 的路徑
Environment="PATH=/var/www/html/SingingVis/venv/bin"
# 啟動指令：使用 3 個工作線程，綁定到 5000 埠
ExecStart=/var/www/html/SingingVis/venv/bin/gunicorn --workers 3 --timeout 600 --bind 0.0.0.0:5000 /flask/singingvis_api_ubuntu:app

[Install]
WantedBy=multi-user.target
```
```sudo systemctl daemon-reload``` <br>
```sudo systemctl start singingvis``` <br>
```sudo systemctl enable singingvis```

### 設定 apache 路由
進入目錄 <br>
```cd /etc/apache2/sites-available``` 

在 000-default.conf 和 default-ssl.conf 貼上以下設定 <br>
```
ProxyPass /api http://127.0.0.1:5000/api
ProxyPassReverse /api http://127.0.0.1:5000/api
ProxyTimeout 600
<Directory /var/www/html/SingingVis/build>
    Options Indexes FollowSymLinks
    AllowOverride All
    Require all granted
    FallbackResource /SingingVis/build/index.html
</Directory>
```

啟用 ssl 和重啟網站 <br>
```sudo a2enmod ssl``` <br>
```sudo a2ensite default-ssl``` <br>
```sudo systemctl restart apache2```

### 其他設定
在 /var/www/html/SingingVis 新增一個資料夾 ```upload_audio```

### 錯誤訊息
如果 youtube 下載出現錯誤，輸入以下指令 <br>
```pip install --upgrade pip``` <br>
```pip install --upgrade yt-dlp``` <br>
