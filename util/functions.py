import numpy as np




def compute_gaussian(x, y, x0, y0, sigma, norm):
    return norm * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))





def generate_images(size=64, sigma=3, intensity=1,x=10,y=10, noisy=False ,sigma_noise = 0.001):
    image = np.zeros((size,size))

    for x0 in range(size):
        for y0 in range(size):
            gaussian = compute_gaussian(x, y, x0, y0, sigma, intensity)
            gaussian = np.array(gaussian)
            image[x0,y0] = gaussian
    if noisy == True : 
        noise = np.random.normal(0, 1, image.shape)
        
        if np.max(noise) != np.min(noise):
            normalized_noise = sigma_noise*(noise-np.min(noise))/(np.max(noise)-np.min(noise))
        else:
            normalized_noise = np.zeros_like(noise)
        
        image = np.clip(image + normalized_noise, 0, 255)
    if np.max(image)!=np.min(image):
        normalised_image = (image-np.min(image))/(np.max(image)-np.min(image))

       
    

    return normalised_image



def spearman_image_correlation(img1, img2):
    """This function aims to compute the SPC between two images"""


    # Step 1: Convert to grayscale
    img1_gray = np.mean(img1, axis=-1) if img1.ndim == 3 else img1
    img2_gray = np.mean(img2, axis=-1) if img2.ndim == 3 else img2

  
    flat_img1 = img1_gray.flatten()
    flat_img2 = img2_gray.flatten()

    # Step 3: Rank the pixel values
    rank_img1 = np.argsort(flat_img1).argsort()
    rank_img2 = np.argsort(flat_img2).argsort()

    # Step 4: Calculate differences
    differences = rank_img1 - rank_img2

    # Step 5: Square the differences
    squared_diff = differences ** 2

    # Step 6: Sum the squared differences
    sum_squared_diff = np.sum(squared_diff)

    # Step 7: Compute Spearman rank correlation coefficient
    n = len(flat_img1)
    rho = (6 * sum_squared_diff) / (n * (n**2 - 1))

    return rho


def divergence_KL_1(img1,img2):
    img1_gray = np.mean(img1, axis=-1) if img1.ndim == 3 else img1
    img2_gray = np.mean(img2, axis=-1) if img2.ndim == 3 else img2
    flat_img1 = img1_gray.flatten()
    flat_img2 = img2_gray.flatten()
    transpose_img1= flat_img1+1
    transpose_img2= flat_img2+1

 
    return np.sum(transpose_img1 * np.log(transpose_img1 / transpose_img2))


def spearman_noise_correlation(dist, epsilon):
    rank_rho = np.argsort(dist)
    rank_epsilon = np.argsort(epsilon)
    differences = rank_rho-rank_epsilon
    squared_diff = differences**2
    sum_squared_diff =np.sum(squared_diff)
    n = len(rank_rho)
    rho = (6 * sum_squared_diff) / (n * (n**2 - 1))
    
    return rho
