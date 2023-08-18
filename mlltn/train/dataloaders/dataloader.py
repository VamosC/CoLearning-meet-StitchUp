class DualLoader():
    # merge two dataloaders

    def __init__(self, loader1, loader2):

        self.loader1 = loader1
        self.loader2 = loader2
        self.sampler = self

    def set_epoch(self, epoch):
        self.loader1.sampler.set_epoch(epoch)
        self.loader2.sampler.set_epoch(epoch)

    def __iter__(self):

        loader2_iter = iter(self.loader2)
        for data1 in self.loader1:
            try:
                data2 = loader2_iter.next()
            except:
                loader2_iter = iter(self.loader2)
                data2 = loader2_iter.next()

            yield (data1, data2)

    def __len__(self):

        return len(self.loader1)
