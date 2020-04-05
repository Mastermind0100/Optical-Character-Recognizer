[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)

# Optical Character Recognition
This is a code that reads an image and predicts what's written in it. 

## Process 
The code first divides the image into multiple segments (each segment contains a single character). On this segment, a pretrained model is executed to predict the character present in the segment. This then outputs the predicted characters as a string

## Setting up the OCR
Let's start by cloning the repository<br>
```bash
$ git clone https://github.com/Mastermind0100/Optical-Character-Recognizer.git
$ cd Optical-Character-Recognizer
```
Great! You are set up with the repository.<br> 
Let's dive into it!

## Output
The Original photo looks like this:
<br/><br/>
![plate1](https://user-images.githubusercontent.com/36445600/60267373-bca10200-9907-11e9-83ae-0a5e7b4ebb4e.jpg)<br/>
<br/><br/>
Mid Processing Output:
<br/><br/>
![up1](https://user-images.githubusercontent.com/36445600/60267398-c75b9700-9907-11e9-8db5-18642455dbff.png)<br/>
<br/><br/>
Final Text Output (Spyder Console):
<br/><br/>
![up2](https://user-images.githubusercontent.com/36445600/60267456-e6f2bf80-9907-11e9-8d8f-df9e9b6221ea.png)

## License
[![License](http://img.shields.io/:license-mit-blue.svg?style=flat)](http://badges.mit-license.org)<br>
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details