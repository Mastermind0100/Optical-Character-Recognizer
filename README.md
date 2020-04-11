[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)

# Optical Character Recognition
This is a code that reads the text present in an image and predicts what's written in it. 

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

## How to Use the OCR
1. Copy the following codes/files into the directory you are using for your project: (Note that if you have just a single code file, you can copy it to this directory instead)
    * ocr.py
    * fmodelwts.h5

2. In your code, add the following lines:
    ```python
    import ocr
    predict(image.extention)
    ```

3. This code will print the text the code detects in the image you gave as input in the function 'predict'.

4. If you want the function to simply return the predicted text and not print it, then make the following changes to Line 78 of the program 'ocr.py':

    ```python
    return final
    ```
    Also, your code needs to accept it in a variable. So the code in Step 2 will change to:
    ```python
    text = predict(image.extention)
    ```
    
* Note that this is a relatively basic OCR. It does not detect spaces for you or segment words in a sentence. While work is under progress for this, you can do some level of image pre-processing to make this work for you.<br>Watch out for further updates!

## Want to train on your own Dataset?

Go ahead! Fire up 'model.py' and use your own dataset. Hopefully the code is self explanatory.
P.S. The Dataset I used was the [NIST](https://s3.amazonaws.com/nist-srd/SD19/by_class.zip) Dataset. Download the 2nd Edition and have fun manually arranging data :)

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