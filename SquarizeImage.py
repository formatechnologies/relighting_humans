import cv2
import numpy as np

class SquarizeImage:
    def __init__(self, img, mask, square_size):
        self.x_trimmed, self.y_trimmed, self.w_trimmed, self.h_trimmed = cv2.boundingRect(mask)

        self.v_padding = int(0.05 * self.w_trimmed)
        self.w_square = 2 * self.v_padding + (self.w_trimmed if self.w_trimmed > self.h_trimmed else self.h_trimmed)

        self.x_square = int(0.5 * (self.w_square - self.w_trimmed))
        self.y_square = self.v_padding

        self.img_square = 255 * np.ones((self.w_square, self.w_square, 3), dtype=np.uint8)
        self.mask_square = np.zeros((self.w_square, self.w_square), dtype=np.uint8)

        self.img_square[self.y_square:self.y_square+self.h_trimmed,self.x_square:self.x_square+self.w_trimmed,:] = img[self.y_trimmed:self.y_trimmed+self.h_trimmed,self.x_trimmed:self.x_trimmed+self.w_trimmed,:]
        self.mask_square[self.y_square:self.y_square+self.h_trimmed,self.x_square:self.x_square+self.w_trimmed] = mask[self.y_trimmed:self.y_trimmed+self.h_trimmed,self.x_trimmed:self.x_trimmed+self.w_trimmed]

        self.square_size = square_size
        self.img_square = cv2.resize(self.img_square, (square_size,square_size))
        self.mask_square = cv2.resize(self.mask_square, (square_size,square_size))

    def get_squared_image(self):
        return self.img_square.copy()
        
    def get_squared_mask(self):
        return self.mask_square.copy()
    
    def replace_image(self, img_orig, mask_orig, img_square):
        new_img = np.zeros_like(img_orig)
        
        if img_square.shape[2] == 9:
            resized_square = np.empty((self.w_square,self.w_square,9), dtype=img_square.dtype)
            resized_square[:,:,0:3] = cv2.resize(img_square[:,:,0:3], (self.w_square,self.w_square))
            resized_square[:,:,3:6] = cv2.resize(img_square[:,:,3:6], (self.w_square,self.w_square))
            resized_square[:,:,6:9] = cv2.resize(img_square[:,:,6:9], (self.w_square,self.w_square))
        else:
            resized_square = cv2.resize(img_square, (self.w_square,self.w_square))
            if len(resized_square.shape) == 2:
                resized_square = resized_square.reshape((resized_square.shape[0], resized_square.shape[1], new_img.shape[2]))
        
#        print('new_img.shape = {}, resized_square.shape = {}'.format(new_img.shape, resized_square.shape))
        
        if len(img_square.shape) == 3:
            new_img[self.y_trimmed:self.y_trimmed+self.h_trimmed,self.x_trimmed:self.x_trimmed+self.w_trimmed,:] = resized_square[self.y_square:self.y_square+self.h_trimmed,self.x_square:self.x_square+self.w_trimmed,:]
            mask = np.repeat(mask_orig.reshape((mask_orig.shape[0], mask_orig.shape[1], 1)), img_square.shape[2], axis=2) / 255.
        else:
            new_img[self.y_trimmed:self.y_trimmed+self.h_trimmed,self.x_trimmed:self.x_trimmed+self.w_trimmed] = resized_square[self.y_square:self.y_square+self.h_trimmed,self.x_square:self.x_square+self.w_trimmed]
            mask = mask_orig / 255.

#        print('mask.shape = {}, new_img.shape = {}, img_orig.shape = {}'.format(mask.shape, new_img.shape, img_orig.shape))
        
        return mask * new_img + (1. - mask) * img_orig

if __name__ == '__main__':
    img = cv2.imread('../user_selected/IMG_2335.jpg', cv2.IMREAD_COLOR)
    mask = cv2.imread('../user_selected/IMG_2335_mask.png', cv2.IMREAD_GRAYSCALE)

    print('img.shape = {}'.format(img.shape))

    si = SquarizeImage(img, mask, 1024)

    img_square = si.get_squared_image()

    cv2.imwrite('img_sqr.png', img_square)
    cv2.imwrite('mask_sqr.png', si.get_squared_mask())
    
    img_square = cv2.cvtColor(img_square, cv2.COLOR_BGR2RGB)
    img_replaced = si.replace_image(img, mask, img_square)
    
    cv2.imwrite('new_img.png', img_replaced)
