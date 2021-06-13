import sys
sys.path.append('../')
from pycore.tikzeng import *

def main():

    namefile = str(sys.argv[0]).split('.')[0]
    botellneck_layer_offset=4
    sum_offset_BasicBlock=2.75
    n_filters=64

    arch = [
        to_head('..'),
        to_cor(),
        to_begin(),
    ]

    height = 128

    arch.append(
        to_Conv("conv_0", "", n_filters, offset="(0,0,0)", height=128, depth=128,
                width=3))
    arch.append(
        to_ReLu("relu_1", "", "(0,0,0)", "(conv_0-east)", height=height, depth=height,
                width=2))
    arch.append(
        to_Pool("pool_2", offset="(0,0,0)", to="(relu_1-east)", height=128 // 2,
                depth=128 // 2, width=2)
    )

    height=128 // 2

    # BasicBlock 1.1
    num_filters=1

    arch.append(
        to_Conv("conv_3", "", n_filters, offset="(2,0,0)", to="(pool_2-east)", height=height, depth=height,
                    width=3*num_filters))
    arch.append(
        to_ReLu("relu_4", "", "(0,0,0)", "(conv_3-east)", height=height, depth=height,
                width=2))
    arch.append(
        to_Conv("conv_5", "", n_filters, offset="(0,0,0)", to="(relu_4-east)", height=height, depth=height,
                    width=3*num_filters))

    arch.append(
        to_ReLu("relu_7", "", "(0,0,0)", "(conv_5-east)", height=height, depth=height,
                width=2))



    arch.append(to_skip("pool_2", "relu_7"), )
    arch.append(
        to_Sum("sum_6", "(2,%f,0)"%sum_offset_BasicBlock, "(pool_2-north)"))

    # BasicBlock 2.2
    # height/=2
    arch.append(
        to_Conv("conv_8", "", n_filters, offset="(2,0,0)", to="(relu_7-east)", height=height, depth=height,
                    width=3*num_filters))
    arch.append(to_connection("relu_7", "conv_8"))

    arch.append(
        to_ReLu("relu_9", "", "(0,0,0)", "(conv_8-east)", height=height, depth=height,
                width=2))

    arch.append(
        to_Conv("conv_10", "", n_filters, offset="(0,0,0)", to="(relu_9-east)", height=height, depth=height,
                width=3*num_filters))
    arch.append(
        to_ReLu("relu_12", "", "(0,0,0)", "(conv_10-east)", height=height, depth=height,
                width=2))

    arch.append(to_skip("relu_7", "conv_10"))
    arch.append(
        to_Sum("sum_11", "(2,%f,0)"%sum_offset_BasicBlock, "(relu_7-north)"))

    num_filters*=2

    # Bottelneck 1
    n_filters*=2

    arch.append(
        to_Conv("conv_13", "", n_filters, offset="(4,0,0)", to="(relu_12-east)", height=height, depth=height,
                width=3 * num_filters))
    arch.append(to_connection("relu_12", "conv_13"))
    arch.append(
        to_ReLu("relu_14", "", "(0,0,0)", "(conv_13-east)", height=height, depth=height,
                width=2))

    arch.append(
        to_Conv("conv_15", "", n_filters, offset="(0,0,0)", to="(relu_14-east)", height=height, depth=height,
                width=3 * num_filters))

    arch.append(
        to_Conv("conv_16", "", n_filters, offset="(%d,%d,0)"%(botellneck_layer_offset, 10), to="(relu_12-north)", height=height, depth=height,
                width=3 * num_filters))

    arch.append(
        to_ReLu("relu_18", "", "(0,0,0)", "(conv_15-east)", height=height, depth=height,
                width=2))



    arch.append(to_bottomskip("relu_12", "conv_16", pos=1.25))
    arch.append(to_topskip("conv_16", "conv_15", pos=1.25))

    arch.append(to_Sum("sum_17", "(1.8,-5,0)", "(conv_16-east)"))

    height/=2
    # BasicBlock
    arch.append(
        to_Conv("conv_19", "", n_filters, offset="(2,0,0)", to="(relu_18-east)", height=height, depth=height,
                width=3 * num_filters))
    arch.append(to_connection("relu_18", "conv_19"))

    arch.append(
        to_ReLu("relu_20", "", "(0,0,0)", "(conv_19-east)", height=height, depth=height,
                width=2))

    arch.append(
        to_Conv("conv_21", "", n_filters, offset="(0,0,0)", to="(relu_20-east)", height=height, depth=height,
                width=3 * num_filters))
    arch.append(
        to_ReLu("relu_23", "", "(0,0,0)", "(conv_21-east)", height=height, depth=height,
                width=2))

    arch.append(to_skip("relu_18", "conv_21"))

    arch.append(to_Sum("sum_8",  "(2.5,%f,0)"%sum_offset_BasicBlock, "(relu_18-north)"))



    # Bottelneck 2
    num_filters*=2
    botellneck_layer_offset+=1
    botellneck_layer_offset_top=6
    n_filters*=2

    arch.append(
        to_Conv("conv_24", "", n_filters, offset="(4,0,0)", to="(relu_23-east)", height=height, depth=height,
                width=3* num_filters))
    arch.append(to_connection("relu_23", "conv_24"))
    arch.append(
        to_ReLu("relu_25", "", "(0,0,0)", "(conv_24-east)", height=height, depth=height,
                width=2))

    arch.append(
        to_Conv("conv_26", "", n_filters, offset="(0,0,0)", to="(relu_25-east)", height=height, depth=height,
                width=3 * num_filters))

    arch.append(
        to_Conv("conv_27", "", n_filters, offset="(%d,%d,0)"%(botellneck_layer_offset, botellneck_layer_offset_top), to="(relu_23-north)", height=height, depth=height,
                width=3 * num_filters))

    arch.append(
        to_ReLu("relu_29", "", "(0,0,0)", "(conv_26-east)", height=height, depth=height,
                width=2))

    arch.append(to_bottomskip("relu_23", "conv_27", pos=1.25))
    arch.append(to_topskip("conv_27", "conv_26", pos=1.25))

    arch.append(to_Sum("sum_28", "(2,-2.5,0)", "(conv_27-east)"))


    # BasicBlock
    arch.append(
        to_Conv("conv_30", "", n_filters, offset="(4,0,0)", to="(relu_29-east)", height=height, depth=height,
                width=3 * num_filters))
    arch.append(to_connection("relu_29", "conv_30"))

    arch.append(
        to_ReLu("relu_31", "", "(0,0,0)", "(conv_30-east)", height=height, depth=height,
                width=2))

    arch.append(
        to_Conv("conv_32", "", n_filters, offset="(0,0,0)", to="(relu_31-east)", height=height, depth=height,
                width=3 * num_filters))
    arch.append(
        to_ReLu("relu_34", "", "(0,0,0)", "(conv_32-east)", height=height, depth=height,
                width=2))

    arch.append(to_skip("relu_29", "conv_32"))
    arch.append(to_Sum("sum_9", "(4.5,%f,0)"%sum_offset_BasicBlock, "(relu_29-north)"))


    # Bottelneck 3
    num_filters *= 2
    botellneck_layer_offset+=1
    botellneck_layer_offset_top=3

    height/=2
    n_filters*=2
    arch.append(
        to_Conv("conv_35", "", n_filters, offset="(4,0,0)", to="(relu_34-east)", height=height, depth=height,
                width=3 * num_filters))
    arch.append(to_connection("relu_34", "conv_35"))
    arch.append(
        to_ReLu("relu_36", "", "(0,0,0)", "(conv_35-east)", height=height, depth=height,
                width=2))

    arch.append(
        to_Conv("conv_37", "", n_filters, offset="(0,0,0)", to="(relu_36-east)", height=height, depth=height,
                width=3 * num_filters))

    arch.append(
        to_Conv("conv_38", "", n_filters, offset="(%d,%d,0)"%(botellneck_layer_offset, botellneck_layer_offset_top), to="(relu_34-north)", height=height, depth=height,
                width=3 * num_filters))

    arch.append(
        to_ReLu("relu_40", "", "(0,0,0)", "(conv_37-east)", height=height, depth=height,
                width=2))

    arch.append(to_bottomskip("relu_34", "conv_38", pos=1.25))
    arch.append(to_topskip("conv_38", "conv_37", pos=1.25))
    arch.append(to_Sum("sum_39", "(3.4,-1.,0)", "(conv_38-east)"))



    # BasicBlock
    arch.append(
        to_Conv("conv_41", "", n_filters, offset="(3,0,0)", to="(relu_40-east)", height=height, depth=height,
                width=3 * num_filters))
    arch.append(to_connection("relu_40", "conv_41"))

    arch.append(
        to_ReLu("relu_42", "", "(0,0,0)", "(conv_41-east)", height=height, depth=height,
                width=2))

    arch.append(
        to_Conv("conv_43", "", n_filters, offset="(0,0,0)", to="(relu_42-east)", height=height, depth=height,
                width=3 * num_filters))
    arch.append(
        to_ReLu("relu_45", "", "(0,0,0)", "(conv_43-east)", height=height, depth=height,
                width=2))

    arch.append(to_skip("relu_40", "conv_43"))
    arch.append(to_Sum("sum_44", "(7,%f,0)"%sum_offset_BasicBlock, "(relu_40-north)"))
    height/=2
    arch.append(
        to_GlobalPool("GlobPool", offset="(2,0,0)", to="(relu_45-east)", height=height/2, depth=height,
                width=3 * num_filters))
    arch.append(to_connection("relu_45", "GlobPool"))


    arch.append(to_Flatten("flatten_1", 512 ,"(1,0,0)", "(GlobPool-east)", height=height*4))

    arch.append(to_connection("GlobPool", "flatten_1"))

    arch.append(to_SoftMax("soft_1", 2 ,"(1,0,0)", "(flatten_1-east)", height=4))
    arch.append(to_connection("flatten_1", "soft_1"))

    # arch.append(to_ReLu("relu2", 1300, "(1,0,0)", "(relu1-east)", depth=48),)
    # arch.append(to_connection("relu1", "relu2"),)
    # #
    # arch.append(to_ReLu("relu3", 50, "(1,0,0)", "(relu2-east)", depth=9),)
    # #
    # arch.append(to_connection("relu2", "relu3"),)
    # #
    # arch.append(to_SoftMax("soft1", 2 ,"(1,0,0)", "(relu3-east)", depth=2),)
    # arch.append(to_connection("relu3", "soft1"),)

    arch.append(to_end())

    to_generate(arch, namefile + '.tex' )

if __name__ == '__main__':
    main()