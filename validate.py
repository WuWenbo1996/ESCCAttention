"""
Validate models by using test data.
"""
import torch

def evaluate(model, test_loader, loss_fn):
	model.eval()

	acc = 0
	total = 0
	loss = 0
	step = 0

	with torch.no_grad():
		for batch_idx, data in enumerate(test_loader):
			inputs = data['spectrogram'].cuda()
			target = data['label'].cuda()

			outputs = model(inputs)

			# Batch Loss
			loss += loss_fn(outputs, target).item()
			step += 1

			# Batch Accuracy
			_, predicted = torch.max(outputs.data, 1)
			acc += (predicted == target).sum().item()
			total += target.size(0)

	acc = acc / total
	loss = loss / step

	return acc, loss
