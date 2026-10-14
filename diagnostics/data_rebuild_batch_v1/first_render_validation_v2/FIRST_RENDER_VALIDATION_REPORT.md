# 首资产透明渲染独立CPU验证

passed=true仅限0086首资产渲染/共同帧几何/真实相机一致性，可继续其余已审资产渲染；不是训练批准或全批质量认证。

8张实际512RGBA完整hash通过，alpha非空/非整幅不透明，强alpha覆盖1.31%..8.81%，各view边界未裁切。首相机screen-right+rawY/up+rawZ，rawlengthY。全部实际相机camera-to-world与inverse一致，独立30deg FOV K、世界点投影审查maxerr5.68e-14pixels。Blender实际float32对角旋转orthogonalityerror2.028e-7，使用1e-6容差；首版1e-7过紧失败已保留，不是camera真实错误。

0/180侧视完整1,495,340三角投影binarymask vsCycles强alpha IoU=.963451/.963649，alpha强像素在几何mask外=0，几何mask多876/871像素，主要二值Pillowinclusive raster vsCycles antialias/subpixel边界差；不要求像素完全一致。没有深度排序或bestfit（union silhouette）。

已直接read_image八图checkercontact，浅银车body/窗/鼻/roof/轮组完整，未见旧合成alpha那样删车身/切轮现象。appearance偏平为渲染光照/材质效果，不认证与历史RGB同等；训练图预处理及其它五资产pixelreview仍需后续检查。无GPU/Blender重渲/模型编码或训练，只CPU只读输入；仅写新validation目录与脚本。v2前台16.819s退出0，没有后台worker。
