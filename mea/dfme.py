# from __future__ import print_function
import argparse

import torch
import torch.optim as optim
import os, random
import time as dfme_time
from sklearn.metrics import accuracy_score

import supervisor
from tools import val_atk
from .dfme_utils import *

from mea import gan
from unlearn import evaluate
from .approximate_gradients import dfme_estimate_gradient_objective

print("torch version", torch.__version__)

def myprint(a):
    print(a)

def student_loss(args, s_logit, t_logit, return_t_logits=False):
    """Kl/ L1 Loss for student"""
    print_logits = False
    if args.loss == "l1":
        loss_fn = F.l1_loss
        loss = loss_fn(s_logit, t_logit.detach())
    elif args.loss == "kl":
        loss_fn = F.kl_div
        s_logit = F.log_softmax(s_logit, dim=1)
        t_logit = F.softmax(t_logit, dim=1)
        loss = loss_fn(s_logit, t_logit.detach(), reduction="batchmean")
    else:
        raise ValueError(args.loss)

    if return_t_logits:
        return loss, t_logit.detach()
    else:
        return loss

def generator_loss(args, s_logit, t_logit, z=None, z_logit=None, reduction="mean"):
    assert 0

    loss = - F.l1_loss(s_logit, t_logit, reduction=reduction)

    return loss


def train(args, teacher, student, generator, device, optimizer, epoch):
    """Main Loop for one epoch of Training Generator and Student"""
    global file
    teacher.eval()
    student.train()

    optimizer_S, optimizer_G = optimizer

    # gradients = []
    x_query = []
    y_query = []
    for i in range(args.epoch_itrs):
        """Repeat epoch_itrs times per epoch"""
        for _ in range(args.g_iter):
            # Sample Random Noise
            z = torch.randn((args.batch_size, args.nz)).to(device)
            optimizer_G.zero_grad()
            generator.train()
            # Get fake image from generator
            fake = generator(z, pre_x=args.approx_grad)  # pre_x returns the output of G before applying the activation
            # print("fake", fake)
            # print("fake.max", torch.max(fake))
            # print("fake.min", torch.min(fake))

            ## APPOX GRADIENT
            approx_grad_wrt_x, loss_G = dfme_estimate_gradient_objective(args, teacher, student, fake,
                                                                    epsilon=args.grad_epsilon, m=args.grad_m,
                                                                    num_classes=args.num_classes,
                                                                    device=device, pre_x=True,
                                                                    dataset='speech_commands')

            fake.backward(approx_grad_wrt_x)
            optimizer_G.step()

            # if i == 0 and args.rec_grad_norm:
            #     x_true_grad = measure_true_grad_norm(args, fake)

        # student
        for _ in range(args.d_iter):
            z = torch.randn((args.batch_size, args.nz)).to(device)
            fake = generator(z).detach()
            optimizer_S.zero_grad()

            with torch.no_grad():
                t_logit = teacher(fake)

            # Correction for the fake logits
            if args.loss == "l1" and args.no_logits:
                t_logit = F.log_softmax(t_logit, dim=1).detach()
                if args.logit_correction == 'min':
                    t_logit -= t_logit.min(dim=1).values.view(-1, 1).detach()
                elif args.logit_correction == 'mean':
                    t_logit -= t_logit.mean(dim=1).view(-1, 1).detach()
            s_logit = student(fake)

            x_query.append(fake.detach().cpu())
            y_query.append(t_logit.cpu())

            loss_S = student_loss(args, s_logit, t_logit)
            loss_S.backward()
            optimizer_S.step()

        # Log Results
        if i % args.log_interval == 0:
            myprint(
                f'Train Epoch: {epoch} [{i}/{args.epoch_itrs} ({100 * float(i) / float(args.epoch_itrs):.0f}%)]\tG_Loss: {loss_G.item():.6f} S_loss: {loss_S.item():.6f}')

            # if i == 0:
                # with open(args.log_dir + "/loss.csv", "a") as f:
                #     f.write("%d,%f,%f\n" % (epoch, loss_G, loss_S))

            if args.rec_grad_norm and i == 0:

                G_grad_norm, S_grad_norm = compute_grad_norms(generator, student)
                # if i == 0:
                #     with open(args.log_dir + "/norm_grad.csv", "a") as f:
                #         f.write("%d,%f,%f,%f\n" % (epoch, G_grad_norm, S_grad_norm, x_true_grad))

        # update query budget
        args.query_budget -= args.cost_per_iteration

        if args.query_budget < args.cost_per_iteration:
            return torch.cat(x_query), torch.cat(y_query)

    return torch.cat(x_query), torch.cat(y_query)


def test(args, student=None, generator=None, device="cuda", test_loader=None, epoch=0):
    global file
    student.eval()
    generator.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = student(data)

            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    myprint('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    with open(args.log_dir + "/accuracy.csv", "a") as f:
        f.write("%d,%f\n" % (epoch, accuracy))
    acc = correct / len(test_loader.dataset)
    return acc


def compute_grad_norms(generator, student):
    G_grad = []
    for n, p in generator.named_parameters():
        if "weight" in n:
            # print('===========\ngradient{}\n----------\n{}'.format(n, p.grad.norm().to("cpu")))
            G_grad.append(p.grad.norm().to("cpu"))

    S_grad = []
    for n, p in student.named_parameters():
        if "weight" in n:
            # print('===========\ngradient{}\n----------\n{}'.format(n, p.grad.norm().to("cpu")))
            S_grad.append(p.grad.norm().to("cpu"))
    return np.mean(G_grad), np.mean(S_grad)

def get_evaluation(
    model,
    wm_model,
    test_loader,
    device,
):
    retain_acc_dict = evaluate(model, test_loader, device)

    model.eval()
    wm_model.eval()
    predictions1 = []
    predictions2 = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)

            outputs1 = wm_model(inputs)
            outputs2 = model(inputs)
            _, predicted1 = torch.max(outputs1, 1)
            _, predicted2 = torch.max(outputs2, 1)

            predictions1.append(predicted1.cpu())
            predictions2.append(predicted2.cpu())
        predictions1 = torch.cat(predictions1).flatten()
        predictions2 = torch.cat(predictions2).flatten()

        fidelity = accuracy_score(predictions1.numpy(), predictions2.numpy()) * 100.

    return retain_acc_dict["Acc"], fidelity


def dfme(student, teacher, test_loader, num_classes, logger, model_path, poison_path=None, dataset='cifar10'):
    # Training settings
    parser = argparse.ArgumentParser(description='DFAD CIFAR')
    parser.add_argument('--batch_size', type=int, default=384, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--query_budget', type=float, default=20, metavar='N',
                        help='Query budget for the extraction attack in millions (default: 20M)')
    parser.add_argument('--epoch_itrs', type=int, default=50)
    parser.add_argument('--g_iter', type=int, default=1, help="Number of generator iterations per epoch_iter")
    parser.add_argument('--d_iter', type=int, default=5, help="Number of discriminator iterations per epoch_iter")

    parser.add_argument('--lr_S', type=float, default=0.05, metavar='LR', help='Student learning rate (default: 0.1)')#TODO
    parser.add_argument('--lr_G', type=float, default=1e-4, help='Generator learning rate (default: 1e-4)')#TODO
    parser.add_argument('--nz', type=int, default=256, help="Size of random noise input to generator")

    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--loss', type=str, default='l1', choices=['l1', 'kl'], )
    parser.add_argument('--scheduler', type=str, default='multistep', choices=['multistep', 'cosine', "none"], )
    parser.add_argument('--steps', nargs='+', default=[0.1, 0.3, 0.5], type=float,
                        help="Percentage epochs at which to take next step")
    parser.add_argument('--scale', type=float, default=3e-1, help="Fractional decrease in lr")

    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=random.randint(0, 100000), metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--student_load_path', type=str, default=None)
    parser.add_argument('--model_id', type=str, default="debug")

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default="results")

    # Gradient approximation parameters
    parser.add_argument('--approx_grad', type=int, default=1, help='Always set to 1')
    parser.add_argument('--grad_m', type=int, default=1, help='Number of steps to approximate the gradients')
    parser.add_argument('--grad_epsilon', type=float, default=1e-3)#TODO

    parser.add_argument('--forward_differences', type=int, default=1, help='Always set to 1')

    # Eigenvalues computation parameters
    parser.add_argument('--no_logits', type=int, default=1)
    parser.add_argument('--logit_correction', type=str, default='mean', choices=['none', 'mean'])

    parser.add_argument('--rec_grad_norm', type=int, default=1)
    parser.add_argument('--MAZE', type=int, default=0)

    # args = parser.parse_args()
    args, unknown = parser.parse_known_args()

    args.query_budget *= 10 ** 6
    args.query_budget = int(args.query_budget)
    if args.MAZE:
        print("\n" * 2)
        print("#### /!\ OVERWRITING ALL PARAMETERS FOR MAZE REPLCIATION ####")
        print("\n" * 2)
        args.scheduer = "cosine"
        args.loss = "kl"
        args.batch_size = 128
        args.g_iter = 1
        args.d_iter = 5
        args.grad_m = 10
        args.lr_G = 1e-4
        args.lr_S = 1e-1

    #logger
    logger.info(args)
    logger.info('-----------Extract Model--------------')
    logger.info('Epoch \t lr \t Time \t ACC \t Fidelity')

    # Prepare the environment
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:%d" % args.device if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    args.device = device
    args.normalization_coefs = None
    args.G_activation = torch.tanh

    args.num_classes = num_classes

    teacher.eval()
    teacher = teacher.to(device)

    correct = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = teacher(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTeacher - Test set: Accuracy: {}/{} ({:.4f}%)\n'.format(correct, len(test_loader.dataset), accuracy))

    with torch.no_grad():
        test_acc, fidelity = get_evaluation(
            student,
            teacher,
            test_loader,
            device,
        )

    logger.info('{} \t {:.3f} \t {:.1f} \t {:.4f} \t {:.4f}'.format(
        0, 0.001, 0, test_acc,
        fidelity))

    if dataset == 'speech_commands':
        generator = gan.GeneratorA(nz=args.nz, nc=1, img_size=32, activation=args.G_activation, data='speech_commands')
        args.lr_S = 0.05
        args.lr_G = 1e-3
        args.grad_epsilon = 1e-3*80
        args.batch_size = 512
        # args.grad_m = 1
    else:
        generator = gan.GeneratorA(nz=args.nz, nc=3, img_size=32, activation=args.G_activation)

    student = student.to(device)
    generator = generator.to(device)

    args.generator = generator
    args.student = student
    args.teacher = teacher

    ## Compute the number of epochs with the given query budget:
    args.cost_per_iteration = args.batch_size * (args.g_iter * (args.grad_m + 1) + args.d_iter)

    number_epochs = args.query_budget // (args.cost_per_iteration * args.epoch_itrs) + 1

    print(f"\nTotal budget: {args.query_budget // 1000}k")
    print("Cost per iterations: ", args.cost_per_iteration)
    print("Total number of epochs: ", number_epochs)

    optimizer_S = optim.SGD(student.parameters(), lr=args.lr_S, weight_decay=args.weight_decay, momentum=0.9)

    if args.MAZE:
        optimizer_G = optim.SGD(generator.parameters(), lr=args.lr_G, weight_decay=args.weight_decay, momentum=0.9)
    else:
        optimizer_G = optim.Adam(generator.parameters(), lr=args.lr_G)

    steps = sorted([int(step * number_epochs) for step in args.steps])
    print("Learning rate scheduling at steps: ", steps)

    if args.scheduler == "multistep":
        scheduler_S = optim.lr_scheduler.MultiStepLR(optimizer_S, steps, args.scale)
        scheduler_G = optim.lr_scheduler.MultiStepLR(optimizer_G, steps, args.scale)
    elif args.scheduler == "cosine":
        scheduler_S = optim.lr_scheduler.CosineAnnealingLR(optimizer_S, number_epochs)
        scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, number_epochs)

    best_acc = 0
    x_query_list = []
    y_query_list = []
    for epoch in range(1, number_epochs + 1):
        # Train

        start_time = dfme_time.perf_counter()
        if args.scheduler != "none":
            scheduler_S.step()
            scheduler_G.step()

        x_query, y_query = train(args, teacher=teacher, student=student, generator=generator, device=device,
              optimizer=[optimizer_S, optimizer_G], epoch=epoch)

        x_query_list.append(x_query)
        y_query_list.append(y_query)

        end_time = dfme_time.perf_counter()
        # Test
        # acc = test(args, student=student, generator=generator, device=device, test_loader=test_loader, epoch=epoch)
        with torch.no_grad():
            test_acc, fidelity = get_evaluation(
                student,
                teacher,
                test_loader,
                device,
            )

        logger.info('{} \t {:.3f} \t {:.1f} \t {:.4f} \t {:.4f}'.format(
            epoch, optimizer_S.param_groups[0]['lr'], end_time - start_time, test_acc,
            fidelity))

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(student.module.state_dict(), model_path)
    if not os.path.exists(os.path.join(poison_path, 'dfme_data')):
        os.makedirs(os.path.join(poison_path, 'dfme_data'))
    torch.save(torch.cat(x_query_list), os.path.join(poison_path, 'dfme_data/x_query.pt'))
    torch.save(torch.cat(y_query_list), os.path.join(poison_path, 'dfme_data/y_query.pt'))

    myprint("Best Acc=%.6f" % best_acc)
    return student
