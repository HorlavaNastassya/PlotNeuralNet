import sys
sys.path.append('../')
from pycore.tikzeng import *



def main():
    namefile = str(sys.argv[0]).split('.')[0]

    arch = [
        to_head('..'),
        to_cor(),
        to_begin(),

    ]
    height, width=3, 15
    arch.append(
        to_Conv("conv_1", "", "Convolution Set", offset="(0,0,0)", height=height, depth=0,
                width=width))
    arch.append(
        to_ReLu("relu_1", "ReLU", "(0.5,0,0)", "(conv_1-east)",  height=height, depth=0,
                width=width))
    #
    arch.append(
        to_Pool("pool_1", offset="(0.5,0,0)", to="(relu_1-east)", height=height, depth=0,
                width=width, caption="MaxPooling3D" )
    )

    arch.append(
        to_GlobalPool("GlobPool", offset="(0.5,0,0)", to="(pool_1-east)", height=height, depth=0,
                width=width, caption="Average Pooling 3D"))

    arch.append(to_Flatten("flatten_1", "Flatten" ,"(0.5,0,0)", "(GlobPool-east)", height=height, depth=0,
                width=width))

    arch.append(to_SoftMax("soft1", "Linear FC", "(0.5,0,0)", "(flatten_1-east)", height=height, depth=0,
                width=width), )

    # arch.append(to_Dropout("dropout_1", "Dropout", "(0.5,0,0)", "(soft1-east)", height=height, depth=0,
    #             width=width))
    #
    # arch.append(
    #     to_SELayer("se_1", "SE block", "(0.5,0,0)", "(dropout_1-east)", height=height, depth=0,
    #             width=width))

    # for i in range(2, 6, 1):
    #     conv_name = "conv_%s" % i
    #     pool_name = "pool_%s" % i
    #     relu_name = "relu_%s" % i
    #     arch.append(
    #         to_Conv(conv_name, "", 8*(2**i), offset="(0,0,0)", to="(pool_%d-east)" % (i - 1), height=128 // (2 ** (i-1)),
    #                 depth=128 // (2 ** (i-1)), width=3*i))
    #     arch.append(
    #         to_ReLu(relu_name, "", "(0,0,0)", "(conv_%d-east)" % i, height=128 // (2 ** (i-1)),
    #                 depth=128 // (2 ** (i-1)),
    #                 width=2))
    #
    #     arch.append(
    #         to_Pool(pool_name, offset="(1,0,0)", to="(relu_%d-east)" % i, height=128 // (2 ** (i)),
    #                 depth=128 // (2 ** (i)), width=1)
    #     )
    #     arch.append(to_connection(relu_name, pool_name))
    #
    # arch.append(to_Flatten("flatten_1", 32256 ,"(1,0,0)", "(pool_5-east)", height=120))
    # arch.append(to_connection("pool_5", "flatten_1"), )
    #
    #
    # arch.append(to_ReLu("relu1", 1300 ,"(1,0,0)", "(flatten_1-east)", height=48))
    # arch.append(to_connection("flatten_1", "relu1"), )
    #
    # arch.append(to_ReLu("relu2", 50, "(1,0,0)", "(relu1-east)", height=15),)
    # arch.append(to_connection("relu1", "relu2"),)
    # #
    # arch.append(to_SoftMax("soft1", 2, "(1,0,0)", "(relu2-east)", height=4),)
    # #
    # arch.append(to_connection("relu2", "soft1"),)
    # #


    arch.append(to_end())

    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()