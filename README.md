# Flask CNN Indonesia News Classification using Logistic Regression

Program ini melatih/train model yang didapat dari hasil crawling yang bisa dilihat pada repo berikut [https://github.com/Ranwiesiel/ppw](/null). Pada aplikasi ini menggunakan algoritma Logistic Regression yang menghasil output 0,1. input menggunakan link/url dari berita CNN ID, dan juga web berita lainnya.

## Getting Started
Sebelum digunakan wajib melakukan beberapa tahapan prasyarat dan penginstalan.

### Prasyarat

Python versi: 3.9+

### Instalasi

1. Clone repo.
```shell
git clone https://github.com/Ranwiesiel/flask-cnn-id-news-classification.git
```
2. Install library.
```python
pip install -r requirements.txt
```
3. Mengganti url remote git untuk menghindari salah push.
```shell
git remote set-url origin github_username/repo_name
git remote -v # confirm the changes
```

## Menjalankan Program

Untuk menjalankan program.
```python
python3 main.py
```

## Demo

Aplikasi dideploy dalam web berikut [https://prediksi-berita.ranwiesiel.serv00.net/](/null)

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request