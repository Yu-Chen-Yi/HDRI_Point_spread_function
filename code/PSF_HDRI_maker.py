from HDRI_functions import *

if __name__ == '__main__':
    smooth_term = 50
    for smooth_term in [0,25,50,75,100,125,150]:
        ##### Test image: folder name #####
        TestImage = 'metalens_PSF'
        print(f'---------- Test Image is {TestImage} ----------')

        ### Whole HDR flow ### 
        print('Start to process HDR flow...')

        # Camera response calibration
        radiance = CameraResponseCalibration(f'../TestImage/{TestImage}', lambda_= smooth_term)
        print('--Camera response calibration done')

        ### Save result ###
        print(f'Saving results..._/{TestImage}')
        SaveImg(radiance, f'../Result/{TestImage}/{TestImage}_lambda_{smooth_term}_HDRI.png')