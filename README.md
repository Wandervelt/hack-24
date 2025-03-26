**CV Hackathon Submission**

This is our submission to the Aaltoes Computer Vision Hackathon. Our team is kaphKo (hack-24). Team participants include Haitham Al-Shami, Rohail Malik, and Hari Prasanth S.M. 

**Overview**
In this project, we participated in a challenge to segment AI tampered regions in images. We attempted an interesting solution using knowledge distillation.

**Approach**
This approach employs a teacher-student architecture for knowledge distillation. The teacher model Takes both the potentially manipulated image AND its original version as input (6 channels total). This gives the teacher access to the true image to better learn manipulation patterns.

The student model Takes only the potentially manipulated image (3 channels). The student needs to identify manipulations without seeing the original. 

Three losses were used:
Boundary Loss: Focuses on detecting the edges/boundaries of manipulated regions.
Dice-BCE-Boundary Loss: Combines Binary Cross-Entropy, Dice coefficient, and Boundary loss to improve segmentation results.
Distillation Loss: Transfers knowledge from teacher to student by making the student match both the ground truth masks and the teacher's predictions.

4. Data Augmentation Techniques
The code uses two key augmentation strategies:

Mixup: Blends two images and their masks to create new training samples.
CutMix: Cuts and pastes regions from different images to create new training samples.

Additional basic transformations were made. 

For training CosineAnnealingLR is implemented to prevent gradient being stuck.
