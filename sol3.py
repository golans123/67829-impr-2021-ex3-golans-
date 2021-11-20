import numpy as np
import matplotlib.pyplot as plt
import imageio
from scipy import ndimage
from scipy import signal
import skimage.color

NUM_SHADES = 256
MAX_SHADE_VAL = 255
GRAY_REPRESENTATION = 1
RGB_REPRESENTATION = 2


# ************ 3.1 Gaussian & Laplacian pyramid construction ****************


def create_filter_vec(filter_size):
    """
    filter_vec is a row vector of shape (1, filter_size) used for the
    pyramid construction. This filter should be built using a consequent 1D
    convolutions of [1 1] with itself in order to derive a row of the binomial
    coefficients which is a good approximation to the Gaussian profile. The
    filter_vec should be normalized.
    :param filter_size:  the size of the Gaussian filter (an odd scalar that
    represents a squared filter) to be used in constructing the pyramid filter
    (e.g for filter_size = 3 you should get [0.25, 0.5, 0.25]). You may assume
    the filter size will be >=2.
    :return: filter_vec
    """
    filter_vec = np.array([1, 1])
    final_filter_vec = np.array([1, 1])
    for i in range(3, filter_size + 1):
        final_filter_vec = \
            signal.convolve(in1=final_filter_vec, in2=filter_vec)
    # normalize
    final_filter_vec = final_filter_vec / final_filter_vec.sum()
    return final_filter_vec.reshape(1, final_filter_vec.shape[0])


def reduce(image, filter_vec):
    """
    reduce: convolve with this filter_vec twice - once as a row vector and then
    as a column vector.

    use the function scipy.ndimage.filters.convolve to apply the filter on the
    image for best results.

    down-sample an image by taking its even indexes (assuming zero-index and of
     course after blurring).
    :param image:
    :param filter_vec: a row vector
    :return:
    """
    # blur
    # as a row vector
    row_filter_vec = filter_vec
    reduced_image = ndimage.filters.convolve(
        input=image, weights=row_filter_vec, output=None, mode='reflect',
        cval=0.0, origin=0)
    # as a column vector
    col_filter_vec = filter_vec.reshape(filter_vec.shape[1], 1)
    reduced_image = ndimage.filters.convolve(
        input=reduced_image, weights=col_filter_vec, output=None,
        mode='reflect', cval=0.0, origin=0)
    # sub-sample
    reduced_image = reduced_image[::2, ::2]
    return reduced_image


def end_of_pyramid(pyr, pyr_max_levels, num_im_rows, num_im_cols):
    """
    check if the number of levels is exceeded
    the input image dimensions are multiples of 2**(max_levels−1).
    :param pyr:
    :param pyr_max_levels:
    :param num_im_rows:
    :param num_im_cols:
    :return:
    """
    # max_levels_exceeded or minimum_dimension_too_small
    return bool(len(pyr) >= pyr_max_levels) or \
        (min(num_im_rows, num_im_cols) < 16)


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    construct a Gaussian pyramid of a given image.

    when performing both the expand and reduce operations you should convolve
    with this filter_vec twice - once as a row vector and then as a column
    vector (for efficiency).

    The pyramid levels should be arranged in order of descending resolution
    s.t. pyr[0] has the resolution of the given input image im.

    The number of levels in the resulting pyramids should be the largest
    possible s.t. max_levels isn’t exceeded and the minimum dimension (height
    or width) of the lowest resolution image in the pyramid is not smaller than
    16. You may assume that the input image dimensions are multiples of
    2**(max_levels−1).

    use the function scipy.ndimage.filters.convolve to apply the filter on the
    image for best results.

    you should down-sample an image by taking its even indexes (assuming
    zero-index and of course after blurring).

    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the Gaussian filter (an odd scalar that
    represents a squared filter) to be used in constructing the pyramid filter
    (e.g for filter_size = 3 you should get [0.25, 0.5, 0.25]). You may assume
    the filter size will be >=2.

    :return:
    1. the resulting pyramid pyr as a standard python array (i.e. a list,
    not numpy’s array) with maximum length of max_levels, where each element of
    the array is a grayscale image.
    2. filter_vec which is a row vector of shape (1, filter_size) used for the
    pyramid construction. This filter should be built using a consequent 1D
    convolutions of [1 1] with itself in order to derive a row of the binomial
    coefficients which is a good approximation to the Gaussian profile. The
    filter_vec should be normalized.
    """
    pyr = []  # output is a python list
    filter_vec = create_filter_vec(filter_size)
    new_im = im
    num_rows = len(new_im)
    num_cols = len(new_im[0])
    while not end_of_pyramid(pyr, max_levels, num_rows, num_cols):
        # insert a level to the top of the pyramid
        pyr.insert(len(pyr), new_im)
        # if not end_of_pyramid(pyr, max_levels, num_rows, num_cols):
        new_im = reduce(new_im, filter_vec)
        num_rows = len(new_im)
        num_cols = len(new_im[0])
    return pyr, filter_vec


def expand(image, filter_vec):
    """
    Also note that to maintain constant brightness in
    the expand operation 2*filter should actually be used in each convolution,
    though the original filter should be returned.

    up-sample by adding zeros in the odd places.
    :param image:
    :param filter_vec
    :return:
    """
    # up-sample by adding zeros in the odd places.
    expanded_image = np.zeros((2 * image.shape[0], 2 * image.shape[1]))
    expanded_image[::2, ::2] = image
    # blur
    # to maintain constant brightness 2*filter should actually be used
    # as a row vector
    row_filter_vec = filter_vec
    expanded_image = ndimage.filters.convolve(
       input=expanded_image, weights=2*row_filter_vec, output=None,
       mode='reflect', cval=0.0, origin=0)  # scipy.signal.convolve2d
    # as a column vector
    col_filter_vec = filter_vec.reshape(filter_vec.shape[1], 1)
    expanded_image = ndimage.filters.convolve(
       input=expanded_image, weights=2*col_filter_vec, output=None,
       mode='reflect', cval=0.0, origin=0)
    return expanded_image


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    construct a Laplacian pyramid of a given image.
    when performing both the expand operation you should convolve
    with this filter_vec twice - once as a row vector and then as a column
    vector (for efficiency).

    The pyramid levels should be arranged in order of descending resolution
    s.t. pyr[0] has the resolution of the given input image im.

    The number of levels in the resulting pyramids should be the largest
    possible s.t. max_levels isn’t exceeded and the minimum dimension (height
    or width) of the lowest resolution image in the pyramid is not smaller than
    16. You may assume that the input image dimensions are multiples of
    2**(max_levels−1).

    use the function scipy.ndimage.filters.convolve to apply the filter on the
    image for best results.

    you should up-sample by adding zeros in the odd places.

    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the Gaussian filter (an odd scalar that
    represents a squared filter) to be used in constructing the pyramid filter
    (e.g for filter_size = 3 you should get [0.25, 0.5, 0.25]). You may assume
    the filter size will be >=2.

    :return:
    1. the resulting pyramid pyr as a standard python array (i.e. a list,
    not numpy’s array) with maximum length of max_levels, where each element of
    the array is a grayscale image.
    2. filter_vec which is row vector of shape (1, filter_size) used for the
    pyramid construction. This filter should be built using a consequent 1D
    convolutions of [1 1] with itself in order to derive a row of the binomial
    coefficients which is a good approximation to the Gaussian profile. The
    filter_vec should be normalized.
    """
    pyr = []  # output is a python list
    gaussian_pyramid, filter_vec = build_gaussian_pyramid(im, max_levels,
                                                          filter_size)
    new_im = None
    for i in range(len(gaussian_pyramid)):
        if i < len(gaussian_pyramid) - 1:
            expanded_image = expand((gaussian_pyramid[i + 1]), filter_vec)
            new_im = (gaussian_pyramid[i]) - expanded_image
        if i == len(gaussian_pyramid) - 1:
            new_im = gaussian_pyramid[-1]
        # insert level to the top
        pyr.insert(len(pyr), new_im)
    return pyr, filter_vec


# ******************* 3.2 Laplacian pyramid reconstruction *******************


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    the reconstruction of an image from its Laplacian Pyramid.

    Before reconstructing the image img you should multiply each level i of
    the laplacian pyramid by its corresponding coefficient coeff[i]. only
    when this list is all ones we get the original image (up to a negligible
    floating error, e.g. maximal absolute difference around 10−12).
    When some values are different than 1 we will get filtering effects

    :param lpyr: the Laplacian pyramid that is generated by the second
    function in 3.1.
    :param filter_vec: the filter that is generated by the second
    function in 3.1.
    :param coeff: a python list. The list length is the same as the number of
     levels in the pyramid lpyr.
    :return: img
    """
    lpyr = np.array(lpyr, dtype=object)
    for i in range(len(lpyr) - 1, -1, -1):
        lpyr[i] = np.multiply(coeff[i], lpyr[i])
    img = lpyr[-1]
    # for i = len(lpyr) - 1, len(lpyr) - 3, ..., 0.
    for i in range(len(lpyr) - 2, -1, -1):
        img = expand(img, filter_vec) + lpyr[i]
    return img


# ************************** 3.3 Pyramid display *****************************


def linear_stretch_image(image):
    image = (image - np.amin(image)) / (np.amax(image) - np.amin(image))
    image = np.array(image)
    return image


def render_pyramid(pyr, levels):
    """
    You should stretch the values of each pyramid level to [0, 1] before
    composing it into the black wide image. Note that you should stretch the
    values of both pyramid types: Gaussian and Laplacian

    :param pyr: either a Gaussian or Laplacian pyramid as defined above.
    :param levels: the number of levels to present in the result ≤ max_levels.
    The number of levels includes the original image.
    :return: a single black image in which the pyramid levels of the given
    pyramid pyr are stacked horizontally (after stretching the values to
    [0, 1]). The function render_pyramid should only return the big image res.
    """
    # res is a single black image
    res_rows = len(pyr[0])
    res_cols = 0
    for i in range(len(pyr)):
        res_cols += len(pyr[i][0])
    res = np.zeros((res_rows, res_cols))
    i = 0
    cols_index = 0
    while bool(i <= levels - 1) and (i < len(pyr)):
        # stretch the values of each pyramid level to [0, 1]
        stretched_pyr = linear_stretch_image(pyr[i])
        res[:len(pyr[i]), cols_index:cols_index + len(pyr[i][0])] = \
            stretched_pyr
        cols_index += len(pyr[i][0])
        i += 1
    return res


def display_pyramid(pyr, levels):
    """
    The function display_pyramid should use render_pyramid to internally render
    and then display the stacked pyramid image using plt.imshow().

    :param pyr: see render_pyramid
    :param levels: see render_pyramid
    :return:
    """
    result = render_pyramid(pyr, levels)
    plt.imshow(result, cmap='gray')  # todo: cmap='gray'?


# ********************* 4 Pyramid Blending ***********************************


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im,
                     filter_size_mask):
    """
    pyramid blending as described in the lecture.

    im1, im2 and mask should all have the same dimensions and that once again
    you can assume that image dimensions are multiples of 2(max_levels−1).

    convert the mask to np.float64, since fractional values should appear while
    constructing the mask’s pyramid.

    Lout should be reconstructed in each level.

    Make sure the output im_blend is a valid grayscale image in the range
    [0, 1], by clipping the result to that range

    :param im1: an input grayscale image to be blended.
    :param im2: an input grayscale image to be blended.
    :param mask: a boolean (dtype == np.bool) mask containing True and False
    representing which parts of im1 and im2 should appear in the resulting
    im_blend. Note that a value of True corresponds to 1, and False corresponds
    to 0.
    :param max_levels: the max_levels parameter you should use when generating
    the Gaussian and Laplacian pyramid
    :param filter_size_im: the size of the Gaussian filter (an odd scalar that
    represents a squared filter) which defining the filter used in the
    construction of the Laplacian pyramids of im1 and im2.
    :param filter_size_mask: the size of the Gaussian filter(an odd scalar that
    represents a squared filter) which defining the filter used in the
    construction of the Gaussian pyramid of mask
    :return:
    """
    # 1. Construct Laplacian pyramids L1 and L2 for the input images im1 and
    # im2, respectively.
    first_laplacian_pyramid, im1_filter_vec = build_laplacian_pyramid(
        im1, max_levels, filter_size_im)
    second_laplacian_pyramid, im2_filter_vec = build_laplacian_pyramid(
        im2, max_levels, filter_size_im)
    # 2. Construct a Gaussian pyramid Gm for the provided mask
    # (convert it first to np.float64).
    mask = np.array(mask).astype(np.float64)
    mask_gaussian_pyramid, mask_filter_vec = build_gaussian_pyramid(
        mask, max_levels, filter_size_mask)
    # 3. Construct the Laplacian pyramid of the blended image for each level k:
    # Lout[k] = Gm[k] · L1[k] + (1 − Gm[k]) · L2[k], pixel-wise multiplication.
    # Lout should be reconstructed in each level.
    output_laplacian_pyramid = []
    for i in range(len(first_laplacian_pyramid)):
        # changes each entry of the matrix
        temp = -1*mask_gaussian_pyramid[i] + 1
        obj1 = np.multiply(
            mask_gaussian_pyramid[i], first_laplacian_pyramid[i])
        obj2 = np.multiply(
            temp, second_laplacian_pyramid[i])
        output_laplacian_pyramid.insert(
            len(output_laplacian_pyramid), obj1 + obj2)
    # 4. Reconstruct the resulting blended image from the Laplacian pyramid
    # Lout (using ones for coefficients).
    coeff = np.ones(len(output_laplacian_pyramid))
    im_blend = laplacian_to_image(
        output_laplacian_pyramid, im1_filter_vec, coeff)
    # Make sure the output im_blend is a valid grayscale image in the range
    # [0, 1], by clipping the result to that range
    im_blend = np.clip(a=im_blend, a_min=0, a_max=1)
    return im_blend


# ****************** 4.1 Your blending examples ******************************
# Don’t forget to include these additional 6 image files (in jpg format) in
# your submission for the scripts to function properly.


def blending_example_helper(im1_filename, im2_filename, mask_filename):
    im1 = read_image(im1_filename, 2)
    im2 = read_image(im2_filename, 2)
    # mask is grayscale
    mask = read_image(mask_filename, 1)
    max_levels = 5
    filter_size_im = 33
    filter_size_mask = 3
    im_blend = np.zeros_like(im1)
    im_blend[:, :, 0] = pyramid_blending(
        im1[:, :, 0], im2[:, :, 0], mask, max_levels, filter_size_im,
        filter_size_mask)
    im_blend[:, :, 1] = pyramid_blending(
        im1[:, :, 1], im2[:, :, 1], mask, max_levels, filter_size_im,
        filter_size_mask)
    im_blend[:, :, 2] = pyramid_blending(
        im1[:, :, 2], im2[:, :, 2], mask, max_levels, filter_size_im,
        filter_size_mask)
    # Display in a single figure with 4 quadrants:
    # the two input images, the mask, and the resulting blended image
    figure, axes_arr = plt.subplots(2, 2)
    axes_arr[0, 0].imshow(im1)
    axes_arr[0, 1].imshow(im2)
    axes_arr[1, 0].imshow(mask, cmap='gray')
    axes_arr[1, 1].imshow(im_blend)
    plt.show()
    return im1, im2, mask, im_blend


def blending_example1():
    """
    performing pyramid blending on sets of image pairs and masks you find nice.

    Display (using plt.imshow()) the two input images, the mask, and the
    resulting blended image in a single figure (you can use plt.subplot() with
    4 quadrants), before returning these objects.
    The examples should present color images (RGB). To generate blended RGB
    images, perform blending on each color channel separately (on red, green
    and blue) and then combine the results into a single image.
    Important: when you load your own images, you must use relative paths
    together with the this function:
    :return: Each function should return the two images (im1 and im2), the mask
    (mask) and the resulting blended image (im_blend).
    """
    im1_filename = "example1_im1.jpg"
    im2_filename = "example1_im2.jpg"
    mask_filename = "example1_mask.jpg"
    im1, im2, mask, im_blend = blending_example_helper(
        im1_filename, im2_filename, mask_filename)
    return im1, im2, mask, im_blend


def blending_example2():
    im1_filename = "C:/Users/Golan/OneDrive/Documents/GitHub/67829_image_processing/ex3-golans/external/meme_always.jpg"
    im2_filename = "C:/Users/Golan/OneDrive/Documents/GitHub/67829_image_processing/ex3-golans/external/meme_fine.jpg"
    mask_filename = "C:/Users/Golan/OneDrive/Documents/GitHub/67829_image_processing/ex3-golans/simple_mask.jpg"
    im1, im2, mask, im_blend = blending_example_helper(
        im1_filename, im2_filename, mask_filename)
    return im1, im2, mask, im_blend


# ************************* from sol1 ***************************************


def read_image(filename, representation):
    """
    a function which reads an image file and converts it into a given
    representation.
    :param filename: the filename of an image on disk (could be grayscale or
    RGB).
    :param representation: a grayscale image (1) or an RGB image (2).
    :return: an image represented by a matrix of type np.float64 with
    intensities (either grayscale or RGB channel intensities)
    normalized to the range [0; 1].

    You will find the function rgb2gray from the module skimage.color useful,
    as well as imread from
    imageio. We won't ask you to convert a grayscale image to RGB.
    """
    image = imageio.imread(filename).astype(np.float64)
    if representation == GRAY_REPRESENTATION:
        image = skimage.color.rgb2gray(image)
    # normalize intensities
    return image / MAX_SHADE_VAL
