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
    for i in range(3, filter_size + 1, 2):
        final_filter_vec = signal.convolve(in1=final_filter_vec, in2=filter_vec)
    # normalize
    final_filter_vec = final_filter_vec / final_filter_vec.sum()
    # print(f"final_filter_vec: {final_filter_vec}")
    return final_filter_vec


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
    row_filter_vec = filter_vec.reshape(filter_vec.shape[0], 1)
    reduced_image = ndimage.filters.convolve(input=image, weights=row_filter_vec, output=None, mode='reflect', cval=0.0,
                                             origin=0)
    # as a column vector
    col_filter_vec = filter_vec.reshape(1, filter_vec.shape[0])
    reduced_image = ndimage.filters.convolve(input=reduced_image, weights=col_filter_vec, output=None, mode='reflect',
                                             cval=0.0, origin=0)
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
    num_rows = len(im)
    num_cols = len(im[0])
    new_im = im
    while not end_of_pyramid(pyr, max_levels, num_rows, num_cols):
        # insert a level to the top of the pyramid
        pyr.insert(len(pyr), new_im)
        if not end_of_pyramid(pyr, max_levels, num_rows, num_cols):
            new_im = reduce(new_im, filter_vec)
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
    # blur
    # to maintain constant brightness 2*filter should actually be used
    # as a row vector
    row_filter_vec = filter_vec.reshape(filter_vec.shape[0], 1)
    expanded_image = ndimage.filters.convolve(
        input=image, weights=2*row_filter_vec, output=None, mode='reflect',
        cval=0.0, origin=0)
    # as a column vector
    col_filter_vec = filter_vec.reshape(1, filter_vec.shape[0])
    expanded_image = ndimage.filters.convolve(
        input=expanded_image, weights=2*col_filter_vec, output=None,
        mode='reflect', cval=0.0, origin=0)
    # up-sample by adding zeros in the odd places
    expanded_image = np.dstack(
        (expanded_image,np.zeros_like(expanded_image))).reshape(
        expanded_image.shape[0],-1)
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
    num_rows = len(im)
    num_cols = len(im[0])
    gaussian_pyramid, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    new_im = im - expand(gaussian_pyramid[0 + 1], filter_vec)
    for i in range(len(gaussian_pyramid)):
        # insert level to the top
        pyr.insert(len(pyr), new_im)
        # should not be entered in the outer-loop's last iteration
        if i < len(gaussian_pyramid) - 1:
            new_im = new_im - expand(gaussian_pyramid[i + 1], filter_vec)
    return pyr, filter_vec


# ******************* 3.2 Laplacian pyramid reconstruction *******************
  def laplacian_to_image(lpyr, filter_vec, coeff):

      """
      the reconstruction of an image from its Laplacian Pyramid

      :param lpyr: the Laplacian pyramid that is generated by the second
      function in 3.1.
      :param filter_vec: the filter that is generated by the second
      function in 3.1.
      :param coeff: a python list. The list length is the same as the number of
       levels in the pyramid lpyr.
      Before reconstructing the image img you should multiply each level i of
      the laplacian pyramid by its corresponding coefficient coeff[i]. only
      when this list is all ones we get the original image (up to a negligible
      floating error, e.g. maximal absolute difference around 10−12).
      When some values are different than 1 we will get filtering effects
      :return:
      """

if __name__ == '__main__':
    monkey = "ex1_presubmit/presubmit_externals/monkey.jpg"
    # plt.imshow(der_im, cmap='gray')
    # plt.show()
    # pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    # pyr, filter_vec = build_laplacian_pyramid(im, max_levels, filter_size)


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
