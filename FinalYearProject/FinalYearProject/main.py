import numpy as np
import pydicom
import random
import showImage
import ECDHKeyGeneration
import steganography
import reverseSteganography
import arnoldCatMap
import reverseArnoldCatMap
import henonMap
import reverseHenonMap
import analysis


def main():
    ka = random.getrandbits(256)
    kb = random.getrandbits(256)
    G = (55066263022277343669578718895168534326250603453777594175500187360389116729240, 
        32670510020758816978083085130507043184471273380659243275938904335757337482424)
    p = pow(2, 256) - pow(2, 32) - pow(2, 9) - pow(2, 8) - pow(2, 7) - pow(2, 6) - pow(2, 4) - pow(2, 0)
    Qa = ECDHKeyGeneration.apply_double_and_add_method(G = G, k = ka, p = p)
    Qb = ECDHKeyGeneration.apply_double_and_add_method(G = G, k = kb, p = p)
    Sa = ECDHKeyGeneration.apply_double_and_add_method(G = Qb, k = ka, p = p)
    Sb = ECDHKeyGeneration.apply_double_and_add_method(G = Qa, k = kb, p = p)
    
    print("Key1: ")
    print(Sa)

    print("\n")

    print("Key2: ")
    print(Sb)
    assert Sa == Sb

  
    k_str = str(Sa[0]) 
    x0 = int(k_str[0:15]) / 10**15
    y0 = int(k_str[15:30]) / 10**15 
    print("x0 =", x0)
    print("y0 =", y0)

    dicom_path = r"C:\Users\shivp\OneDrive\Desktop\sathishheadscan.dcm"
    dicom_data = pydicom.dcmread(dicom_path) 
    name=str(dicom_data.PatientName)
    dicom_image = dicom_data.pixel_array
    bit_size =dicom_data.BitsStored
    showImage.show_image(dicom_image)
    max_value=pow(2, bit_size) 

    message="Patient Name: "
    message=message+name
    delimiter="####"
    message+=delimiter
    required_space=len(message)*4
    capacity=dicom_image.shape[0]*dicom_image.shape[1]
    msgBits=''.join([format(ord(i),'08b') for i in message])

    stegoImage=steganography.stegono(dicom_image,msgBits,required_space,bit_size)
    showImage.show_image(stegoImage)
    scrambled_image = arnoldCatMap.arnold_cat_map(dicom_image, 15)
    showImage.show_image(scrambled_image)
    diffused_image=henonMap.hennon_map(scrambled_image,scrambled_image.shape[0],max_value,x0,y0)
    showImage.show_image(diffused_image)
    dicom_data.PixelData = diffused_image.tobytes()
    dicom_data.save_as("encrypted_output.dcm")
    rev_hen_image=reverseHenonMap.inverse_hennon_map(diffused_image,diffused_image.shape[0],max_value,x0,y0)
    showImage.show_image(rev_hen_image)
    rev_arn_image=reverseArnoldCatMap.inverse_arnold_cat_map(rev_hen_image,15)
    showImage.show_image(rev_arn_image) 
    # hidden_data=reverseSteganography.revStego(rev_arn_image,bit_size)
    # print(hidden_data)

    analysis.analysisFunc(dicom_image,scrambled_image,diffused_image,rev_arn_image)

main()