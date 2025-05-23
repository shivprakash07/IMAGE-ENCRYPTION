def stegono(image,msgBits,required_space,bit_size):
    size=image.shape[0]

    for i in range(0,required_space):
        pixel_value=image[i//size,i%size]
        pixel_value=format(pixel_value,'0'+str(bit_size)+'b')
        pixel_value=pixel_value[:bit_size-2]+msgBits[i*2:i*2+2]
        pixel_value=int(pixel_value,2)
        image[i//size,i%size]=pixel_value

    return image