const express = require("express");

const router = express.Router();

const {
    signup,
    login,
    getDashboard
} = require("../controllers/ngoController");

const authMiddleware = require("../middleware/authMiddleware");


// NGO Signup Route
router.post("/signup", signup);


// NGO Login Route
router.post("/login", login);


// NGO Dashboard Route
router.get("/dashboard", authMiddleware, getDashboard);


module.exports = router;