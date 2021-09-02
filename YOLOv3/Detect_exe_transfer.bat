cd c:\rl_timberman\yolov3
call "%userprofile%\\anaconda3\\scripts\\activate.bat" ai & python Transfer_learning.py --epsilon 0.999 --epsilon_discount 0.001 --learning_rate 0.001 --node 32 --step_mode 1 --batch_size 8
