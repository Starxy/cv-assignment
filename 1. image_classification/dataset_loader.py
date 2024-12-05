"""
数据集加载工具

@Author: DONG Jixing
@Date: 2024-11-10
"""

from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

class DatasetLoader:
    """通用数据集加载器类,用于加载和预处理常用的图像数据集"""
    def __init__(self, dataset_name, root='./data', batch_size=64, 
                 num_workers=2, pin_memory=True):
        """
        初始化数据集加载器
        Args:
            dataset_name: 数据集名称,如'FashionMNIST','MNIST'等
            root: 数据集保存路径
            batch_size: 批次大小
            num_workers: 数据加载的工作进程数
            pin_memory: 是否将数据加载到CUDA固定内存中
        """
        self.dataset_name = dataset_name
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # 获取数据集的均值和标准差
        mean, std = self._get_mean_std()
        
        # 设置默认的数据转换
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 将图像转换为张量
            transforms.Normalize(mean, std)  # 使用数据集特定的均值和标准差进行标准化
        ])
        
    def set_transform(self, transform):
        """
        设置自定义的数据转换
        Args:
            transform: 自定义的torchvision转换器
        """
        self.transform = transform
        
    def get_dataset(self):
        """
        获取指定的数据集
        Returns:
            trainset: 训练数据集
            testset: 测试数据集
        """
        dataset_dict = {
            'FashionMNIST': torchvision.datasets.FashionMNIST,
            'MNIST': torchvision.datasets.MNIST,
            'CIFAR10': torchvision.datasets.CIFAR10,
            'CIFAR100': torchvision.datasets.CIFAR100
        }
        
        if self.dataset_name not in dataset_dict:
            raise ValueError(f"不支持的数据集: {self.dataset_name}")
            
        dataset_class = dataset_dict[self.dataset_name]
        
        # 加载训练集和测试集
        trainset = dataset_class(
            root=self.root,
            train=True,
            download=True,
            transform=self.transform
        )
        
        testset = dataset_class(
            root=self.root,
            train=False,
            download=True,
            transform=self.transform
        )
        
        return trainset, testset
    
    def get_data_loaders(self):
        """
        获取数据加载器
        Returns:
            train_loader: 训练数据加载器
            test_loader: 测试数据加载器
        """
        trainset, testset = self.get_dataset()
        
        train_loader = DataLoader(
            trainset,
            batch_size=self.batch_size,
            shuffle=True,  # 训练时打乱数据
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        test_loader = DataLoader(
            testset,
            batch_size=self.batch_size,
            shuffle=False,  # 测试时不打乱数据
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        
        return train_loader, test_loader
    
    def _get_mean_std(self):
        """
        获取数据集的均值和标准差
        Returns:
            tuple: (均值,标准差)的元组
        """
        dataset_stats = {
            'FashionMNIST': ((0.5,), (0.5,)),
            'MNIST': ((0.1307,), (0.3081,)),
            'CIFAR10': ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            'CIFAR100': ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        }
        return dataset_stats.get(self.dataset_name, ((0.5,), (0.5,))) 