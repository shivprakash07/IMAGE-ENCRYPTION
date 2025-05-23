def revStego(image,bit_size):
    size=image.shape[0]
    msg=''
    i=0
    while(1):
        char=''
        for j in range (0,4):
            pixel_value=image[(i*4+j)//size,(i*4+j)%size]
            pixel_value=format(pixel_value,'0'+str(bit_size)+'b')
            char=char+(pixel_value[bit_size-2:])
        msgChar=chr(int(char,2))
        msg=msg+msgChar
        if(len(msg)>=4):
            if(msg[len(msg)-4:]=='####'):
                break
        i=i+1
    return msg[:len(msg)-4]  