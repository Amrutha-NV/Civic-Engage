const NGO = require("../models/NGO");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");

// NGO Signup
const signup = async (req, res) => {
    try {

        const { ngoName, email, phone, address, password } = req.body;

        if (!ngoName || !email || !phone || !address || !password) {
            return res.status(400).json({
                message: "Please fill all the fields"
            });
        }

        const existingNGO = await NGO.findOne({ email });

        if (existingNGO) {
            return res.status(400).json({
                message: "NGO already registered"
            });
        }

        const hashedPassword = await bcrypt.hash(password, 10);

        const ngo = new NGO({
            ngoName,
            email,
            phone,
            address,
            password: hashedPassword
        });

        await ngo.save();

        return res.status(201).json({
            success: true,
            message: "NGO Registered Successfully"
        });

    } catch (error) {
        console.log(error);

        return res.status(500).json({
            success: false,
            message: "Server Error"
        });
    }
};


// NGO Login
const login = async (req, res) => {
    try {

        const { email, password } = req.body;

        if (!email || !password) {
            return res.status(400).json({
                message: "Please fill all the fields"
            });
        }


        const ngo = await NGO.findOne({ email });

        if (!ngo) {
            return res.status(404).json({
                message: "NGO not found"
            });
        }


        const isMatch = await bcrypt.compare(password, ngo.password);

        console.log("Entered Password:", password);
console.log("Stored Hash:", ngo.password);
console.log("Password Match:", isMatch);

        if (!isMatch) {
            return res.status(401).json({
                message: "Invalid Password"
            });
        }


        const token = jwt.sign(
            { id: ngo._id },
            process.env.JWT_SECRET,
            { expiresIn: "1d" }
        );


        return res.status(200).json({
            success: true,
            message: "Login Successful",
            token
        });


    } catch (error) {
        console.log(error);

        return res.status(500).json({
            success: false,
            message: "Server Error"
        });
    }
};


// NGO Dashboard
const getDashboard = async (req, res) => {

    try {

        const ngo = await NGO.findById(req.ngo.id)
            .select("-password");


        if (!ngo) {
            return res.status(404).json({
                message: "NGO not found"
            });
        }


        return res.status(200).json({
            success: true,
            dashboard: ngo
        });


    } catch (error) {

        console.log(error);

        return res.status(500).json({
            message: "Server Error"
        });

    }

};


module.exports = {
    signup,
    login,
    getDashboard
};