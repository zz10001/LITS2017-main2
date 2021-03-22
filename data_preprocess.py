import os
import numpy as np
import SimpleITK as sitk

images_path = './LITS/images'
labels_path = './LITS/labels'
if not os.path.exists(images_path):
    print("images_path 不存在")
if not os.path.exists(labels_path):
    print("labels_path 不存在")

output_trainImage = './data/train_image'
output_trainMask = './data/train_mask'
output_testImage = './data/test_image'
output_testMask = './data/test_mask'

trainImage = output_trainImage
trainMask = output_trainMask
if not os.path.exists(trainImage):
    os.makedirs(output_trainImage)
    print("trainImage 输出目录创建成功")
if not os.path.exists(trainMask):
    os.makedirs(trainMask)
    print("trainMask 输出目录创建成功")

testImage = output_testImage
testMask = output_testMask
if not os.path.exists(testImage):
    os.makedirs(testImage)
    print("testImage 输出目录创建成功")
if not os.path.exists(testMask):
    os.makedirs(testMask)
    print("testMask 输出目录创建成功")

BLOCKSIZE = (96, 128, 160) #每个分块的大小
#处理训练数据
MOVESIZE_Z = 30 #Z方向分块移动的步长
MOVESIZE_XY = 64 #XY方向分块移动的步长
TRAIN_TEXT_RATIO = 0.7 # 训练集占总数据的百分比，剩余为测试集

def normalize(slice, bottom=99, down=1):
    """
    normalize image with mean and std for regionnonzero,and clip the value into range
    :param slice:
    :param bottom:
    :param down:
    :return:
    """
    b = np.percentile(slice, bottom)
    t = np.percentile(slice, down)
    slice = np.clip(slice, t, b)

    image_nonzero = slice[np.nonzero(slice)]
    if np.std(slice) == 0 or np.std(image_nonzero) == 0:
        return slice
    else:
        tmp = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        # since the range of intensities is between 0 and 5000 ,
        # the min in the normalized slice corresponds to 0 intensity in unnormalized slice
        # the min is replaced with -9 just to keep track of 0 intensities
        # so that we can discard those intensities afterwards when sampling random patches
        tmp[tmp == tmp.min()] = -9
        return tmp

for file in os.listdir(images_path):
    print(os.path.join(images_path,file),os.path.join(labels_path,file))
    # 1、读取数据  
    image_src = sitk.ReadImage(os.path.join(images_path,file), sitk.sitkInt16)
    mask_src = sitk.ReadImage(os.path.join(labels_path,file.replace('volume','segmentation')), sitk.sitkUInt8)
    image_array = sitk.GetArrayFromImage(image_src)
    mask_array = sitk.GetArrayFromImage(mask_src)
    
    # 2、对image分别进行标准化 
    image_array_nor = normalize(image_array)   

    # 3、人工加入的切片，使得z轴数据为BLOCKSIZE[0]的整数倍
    z_size = 0
    z_add = 0 #需要加入的片数
    while True:
        z_size = z_size + BLOCKSIZE[0]
        if( z_size > image_array.shape[0]):
            z_add = z_size - image_array.shape[0]
            break
    myblackslice = np.ones([512,512]) * int(-9) #黑色区域
    zeromaskslice = np.zeros([512,512])
    for i in range(z_add):
        image_array_nor = np.insert(image_array_nor,image_array_nor.shape[0],myblackslice,axis = 0) #往z轴最后面加入切片
        mask_array = np.insert(mask_array,mask_array.shape[0],zeromaskslice,axis = 0)
    
    # 4、分块处理    训练集不用注释，用这些   
    patch_block_size = BLOCKSIZE
    numberxy = MOVESIZE_XY
    numberz = MOVESIZE_Z   #z方向上每移动numberz，就取一个BLOCKSIZE块    #patch_block_size[0]
    
    width = np.shape(image_array_nor)[1] 
    height = np.shape(image_array_nor)[2] 
    imagez = np.shape(image_array_nor)[0]
    
    block_width = np.array(patch_block_size)[1]
    block_height = np.array(patch_block_size)[2]
    blockz = np.array(patch_block_size)[0]
    
    stridewidth = (width - block_width) // numberxy
    strideheight = (height - block_height) // numberxy
    stridez = (imagez - blockz) // numberz
    
    step_width = width - (stridewidth * numberxy + block_width)
    step_width = step_width // 2
    step_height = height - (strideheight * numberxy + block_height)
    step_height = step_height // 2
    step_z = imagez - (stridez * numberz + blockz)
    step_z = step_z // 2
                    
    image_array_nor_list = []
    mask_array_list = []
    patchnum = []
    
    print(image_array_nor.shape)
    for z in range(step_z, numberz * (stridez + 1) + step_z, numberz):
        for x in range(step_width, numberxy * (stridewidth + 1) + step_width, numberxy):
            for y in range(step_height, numberxy * (strideheight + 1) + step_height, numberxy):
                if np.max(mask_array[z:z + blockz, x:x + block_width, y:y + block_height]) != 0: # 某个分块未进行打标签的将丢弃
                    #print("切%d"%z)
                    patchnum.append(str(z) + str('_') + str(x)+ str('_') + str(y))
                    image_array_nor_list.append(image_array_nor[z:z + blockz, x:x + block_width, y:y + block_height])
                    mask_array_list.append(mask_array[z:z + blockz, x:x + block_width, y:y + block_height])
    
    image_last_list = np.array(image_array_nor_list).reshape((len(image_array_nor_list), blockz, block_width, block_height))
    mask_last_list = np.array(mask_array_list).reshape((len(mask_array_list), blockz, block_width, block_height))
    
    samples, imagez, height, width = np.shape(image_last_list)[0], np.shape(image_last_list)[1], \
                                    np.shape(image_last_list)[2], np.shape(image_last_list)[3]
    
    #print(samples)
    #保存
    for j in range(samples):
        train_image_path = trainImage + "/" + str(file.split('.')[0]) + "_patch_" + str(patchnum[j]) + ".npy"
        train_mask_path = trainMask + "/" + str(file.replace('volume','segmentation').split('.')[0]) + "_patch_" + str(patchnum[j]) + ".npy"
        np.save(train_image_path, image_last_list[j,:,:,:])
        np.save(train_mask_path, mask_last_list[j,:,:,:])
    

    # 训练集不用注释，用这些 

    # 测试集生成代码数据
    # patch_block_size = BLOCKSIZE
    # numberxy = BLOCKSIZE[1]
    # numberz = BLOCKSIZE[0]   
    
    # width = np.shape(image_array_nor)[1] 
    # height = np.shape(image_array_nor)[2] 
    # imagez = np.shape(image_array_nor)[0]
    
    # block_width = np.array(patch_block_size)[1]
    # block_height = np.array(patch_block_size)[2]
    # blockz = np.array(patch_block_size)[0]
    
    # stridewidth = (width - block_width) // numberxy
    # strideheight = (height - block_height) // numberxy
    # stridez = (imagez - blockz) // numberz
    
    # step_width = width - (stridewidth * numberxy + block_width)
    # step_width = step_width // 2
    # step_height = height - (strideheight * numberxy + block_height)
    # step_height = step_height // 2
    # step_z = imagez - (stridez * numberz + blockz)
    # step_z = step_z // 2
                    
    # image_array_nor_list = []
    # mask_array_list = []
    # patchnum = []
    
    # print(image_array_nor.shape)
    # for z in range(step_z, numberz * (stridez + 1) + step_z, numberz):
    #     for x in range(step_width, numberxy * (stridewidth + 1) + step_width, numberxy):
    #         for y in range(step_height, numberxy * (strideheight + 1) + step_height, numberxy):
    #             #if np.max(mask_array[z:z + blockz, x:x + block_width, y:y + block_height]) != 0: # 某个分块未进行打标签的将丢弃
    #             #print("切%d"%z)
    #             patchnum.append(str(z) + str('_') + str(x)+ str('_') + str(y))
    #             image_array_nor_list.append(image_array_nor[z:z + blockz, x:x + block_width, y:y + block_height])
    #             mask_array_list.append(mask_array[z:z + blockz, x:x + block_width, y:y + block_height])
    
    # image_last_list = np.array(image_array_nor_list).reshape((len(image_array_nor_list), blockz, block_width, block_height))
    # mask_last_list = np.array(mask_array_list).reshape((len(mask_array_list), blockz, block_width, block_height))
    
    # samples, imagez, height, width = np.shape(image_last_list)[0], np.shape(image_last_list)[1], \
    #                                 np.shape(image_last_list)[2], np.shape(image_last_list)[3]
    
    # #保存
    # for j in range(samples):
    #     test_image_path = testImage + "/" + str(file.split('.')[0])  + "_add_" + str(z_add) + "_patch_" + str(patchnum[j]) + ".npy"
    #     test_mask_path = testMask + "/" + str(file.split('.')[0]) + "_add_" + str(z_add) + "_patch_" + str(patchnum[j]) + ".npy"
    #     np.save(test_image_path, image_last_list[j,:,:,:])
    #     np.save(test_mask_path, mask_last_list[j,:,:,:])