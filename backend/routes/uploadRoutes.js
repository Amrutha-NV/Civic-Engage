const express = require("express");
const router = express.Router();
const multer = require("multer");
const { uploadImage } = require("../controllers/uploadController");

// Use memory storage for multer
const storage = multer.memoryStorage();
const upload = multer({
  storage,
  limits: { fileSize: 5 * 1024 * 1024 }, // 5MB limit
});

router.post("/upload", upload.single("image"), uploadImage);

module.exports = router;
