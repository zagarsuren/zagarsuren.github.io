# Personal Portfolio Website


## 🚀 Local Development

To run the portfolio locally:

```bash
npm install
npm run dev
```

This will start the Vite development server. Open the browser at:

```
http://localhost:5173
```

Any saved changes in the project files will automatically refresh the browser.

---

## 🚢 Manual Deployment to GitHub Pages

This project uses [`gh-pages`](https://www.npmjs.com/package/gh-pages) for manual deployment.


### 📦 To deploy:

```bash
npm run deploy
git add .
git commit -m "Deploy from main"
git push origin main         
```

This will:

* Build the production version of the site
* Push the `dist/` folder to the `gh-pages` branch
* Automatically publish the site at:

🔗 [https://zagarsuren.github.io](https://zagarsuren.github.io)

---

## 🛠 Tips
* GitHub Pages should be set to serve from the `gh-pages` branch, root folder.

---

## 📁 Project Stack

* ⚛️ React + Vite
* 🎨 Tailwind CSS
* 🎞 Framer Motion
* 🌐 Deployed to GitHub Pages