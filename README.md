# molgen

## Usage
```python

from rdkit import Chem
from model import MolGen

# load data
data = []
with open('qm9.csv', "r") as f:
    for line in f.readlines()[1:]:
        data.append(line.split(",")[1])



# create model
gan_mol = MolGen(data, hidden_dim=128, lr=1e-3, device="cuda")

# create dataloader
loader = model.create_dataloader(data, batch_size=64, shuffle=True, num_workers=10)

# train model for 10000 steps
model.train_n_steps(loader, max_step=10000, evaluate_every=200)

# After training
# generate Smiles molecules
smiles_list = model.generate_n(8)

# convert with rdkit
mol_list = [Chem.MolFromSmiles(m) for m in smiles_list]

# draw
Chem.Draw.MolsToGridImage(mol_list, molsPerRow=4, subImgSize=(250, 250), maxMols=10)

```

## Reference:
@inproceedings{dAutume2019TrainingLG,
  title={Training language GANs from Scratch},
  author={Cyprien de Masson d'Autume and Mihaela Rosca and Jack W. Rae and S. Mohamed},
  booktitle={NeurIPS},
  year={2019}
}