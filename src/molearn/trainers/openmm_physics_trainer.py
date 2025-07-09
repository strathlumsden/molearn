import torch
import torch.nn.functional as F
from molearn.loss_functions import openmm_energy
from molearn.loss_functions.geometric import compute_distance_loss, compute_torsion_loss
from .trainer import Trainer
import os


soft_xml_script = """\
<ForceField>
 <Script>
import openmm as mm
nb = mm.CustomNonbondedForce('C/((r/0.2)^4+1)')
nb.addGlobalParameter('C', 1.0)
sys.addForce(nb)
for i in range(sys.getNumParticles()):
    nb.addParticle([])
exclusions = set()
for bond in data.bonds:
    exclusions.add((min(bond.atom1, bond.atom2), max(bond.atom1, bond.atom2)))
for angle in data.angles:
    exclusions.add((min(angle[0], angle[2]), max(angle[0], angle[2])))
for a1, a2 in exclusions:
    nb.addExclusion(a1, a2)
 </Script>
</ForceField>
"""


class OpenMM_Physics_Trainer(Trainer):
    """
    OpenMM_Physics_Trainer subclasses Trainer and replaces the valid_step and train_step.
    An extra 'physics_loss' is calculated using OpenMM and the forces are inserted into backwards pass.
    To use this trainer requires the additional step of calling :func:`prepare_physics <molearn.trainers.OpenMM_Physics_Trainer.prepare_physics>`.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def prepare_physics(
        self,
        physics_scaling_factor=0.1,
        clamp_threshold=1e8,
        clamp=False,
        start_physics_at=10,
        xml_file=None,
        soft_NB=True,
        **kwargs,
    ):
        """
        Create ``self.physics_loss`` object from :func:`loss_functions.openmm_energy <molearn.loss_functions.openmm_energy>`
        Needs ``self.mol``, ``self.std``, and ``self._data.atoms`` to have been set with :func:`Trainer.set_data<molearn.trainer.Trainer.set_data>`

        :param float physics_scaling_factor: scaling factor saved to ``self.psf`` that is used in :func:`train_step <molearn.trainers.OpenMM_Physics_Trainer.train_step>`. Defaults to 0.1
        :param float clamp_threshold: if ``clamp=True`` is passed then forces will be clamped between -clamp_threshold and clamp_threshold. Default: 1e-8
        :param bool clamp: Whether to clamp the forces. Defaults to False
        :param int start_physics_at: At which epoch the physics loss will be added to the loss. Default: 10
        :param \*\*kwargs: All aditional kwargs will be passed to :func:`openmm_energy <molearn.loss_functions.openmm_energy>`

        """
        if xml_file is None and soft_NB:
            print("using soft nonbonded forces by default")
            from molearn.utils import random_string

            tmp_filename = f"soft_nonbonded_{random_string()}.xml"
            with open(tmp_filename, "w") as f:
                f.write(soft_xml_script)
            xml_file = ["amber14-all.xml", tmp_filename]
            kwargs["remove_NB"] = True
        elif xml_file is None:
            xml_file = ["amber14-all.xml"]
        self.start_physics_at = start_physics_at
        self.psf = physics_scaling_factor
        if clamp:
            clamp_kwargs = dict(max=clamp_threshold, min=-clamp_threshold)
        else:
            clamp_kwargs = None
        
        # Account for the un-normalized transformer data
        std_to_use = 1.0 if self.for_transformer==True else self.std

        self.physics_loss = openmm_energy(
            self.mol,
            std_to_use,
            clamp=clamp_kwargs,
            platform="CUDA" if self.device == torch.device("cuda") else "Reference",
            atoms=self._data.atoms,
            xml_file=xml_file,
            **kwargs,
        )
        os.remove(tmp_filename)

    def common_physics_step(self, batch, latent):
        """
        Called from both :func:`train_step <molearn.trainers.OpenMM_Physics_Trainer.train_step>` and :func:`valid_step <molearn.trainers.OpenMM_Physics_Trainer.valid_step>`.
        Takes random interpolations between adjacent samples latent vectors. These are decoded (decoded structures saved as ``self._internal['generated'] = generated if needed elsewhere) and the energy terms calculated with ``self.physics_loss``.

        :param torch.Tensor batch: tensor of shape [batch_size, 3, n_atoms]. Give access to the mini-batch of structures. This is used to determine ``n_atoms``
        :param torch.Tensor latent: tensor shape [batch_size, 2, 1]. Pass the encoded vectors of the mini-batch.
        """
        alpha = torch.rand(int(len(batch) // 2), 1, 1).type_as(latent)
        latent_interpolated = (1 - alpha) * latent[:-1:2] + alpha * latent[1::2]

        generated = self.autoencoder.decode(latent_interpolated)[:, :, : batch.size(2)]
        self._internal["generated"] = generated
        energy = self.physics_loss(generated)
        energy[energy.isinf()] = 1e35
        energy = torch.clamp(energy, max=1e34)
        energy = energy.nanmean()

        return {
            "physics_loss": energy
        }  # a if not energy.isinf() else torch.tensor(0.0)}
    
    def _compute_transformer_physics_loss(self, predicted_coords: torch.Tensor) -> torch.Tensor:
        """
        Contains the logic for calculating physics loss for a transformer-based architecture.
        Handles reshaping and clamping.
        """
        B, L, N_atoms_per_res, _ = predicted_coords.shape
        
        # Reshape from (B, L, 4, 3) to (B, L*4, 3)
        coords_reshaped = predicted_coords.reshape(B, L * N_atoms_per_res, 3)
        
        # Permute from (B, N, 3) to (B, 3, N) as expected by the openmm_energy module
        coords_permuted = coords_reshaped.permute(0, 2, 1)
        
        # Calculate the energy
        energy = self.physics_loss(coords_permuted)
        
        # Apply clamping and safe averaging for stability
        energy[energy.isinf()] = 1e35
        energy = torch.clamp(energy, max=1e34)
        
        return energy.nanmean()


    def train_step(self, batch):
        """
        This method overrides :func:`Trainer.train_step <molearn.trainers.Trainer.train_step>` and adds an additional 'Physics_loss' term.
        Called from :func:`Trainer.train_epoch <molearn.trainers.Trainer.train_epoch>`.

        :param torch.Tensor batch: tensor shape [Batch size, 3, Number of Atoms]. A mini-batch of protein frames normalised. To recover original data multiple by ``self.std``.
        :returns: Return loss. The dictionary must contain an entry with key ``'loss'`` that :func:`self.train_epoch <molearn.trainers.Trainer.train_epoch>` will call ``result['loss'].backwards()`` to obtain gradients.
        :rtype: dict
        """

        if self.for_transformer:

            true_coords = batch
            predicted_coords = self.autoencoder(true_coords)
            
            # Calculate individual loss components
            mse_loss = F.mse_loss(predicted_coords, true_coords)
            dist_loss = compute_distance_loss(predicted_coords, true_coords)
            torsion_loss = compute_torsion_loss(predicted_coords, true_coords)
            
            results = {
                "mse_loss": mse_loss,
                "dist_loss": dist_loss,
                "torsion_loss": torsion_loss,
            }

            # Add physics loss only after the specified startup epoch
            if self.epoch >= self.start_physics_at:
                physics_loss = self._compute_transformer_physics_loss(predicted_coords)
                results["physics_loss"] = physics_loss
            else:
                results["physics_loss"] = torch.tensor(0.0, device=self.device)

           
            # Use loss_weights as base scaling factors
            s = self.loss_weights
            scale_dist = s['w_dist'] * results['mse_loss'].detach() / (results['dist_loss'].detach() + 1e-8)
            scale_torsion = s['w_torsion'] * results['mse_loss'].detach()/ (results['torsion_loss'].detach() + 1e-8)
           
           # Use a ternary expression for physics scale to avoid division by zero if physics loss is zero
            phys_loss_detached = results['physics_loss'].detach()
            scale_phys = (s['w_phys'] * results['mse_loss'].detach() / (phys_loss_detached + 1e-8)) if phys_loss_detached > 0 else 0.0
            
            total_loss = (
                results['mse_loss'] + 
                scale_dist * results['dist_loss'] + 
                scale_torsion * results['torsion_loss'] +
                scale_phys * results['physics_loss']
            )
                
            results["loss"] = total_loss
            return results

        else:

            results = self.common_step(batch)
            results.update(self.common_physics_step(batch, self._internal["encoded"]))
            
            with torch.no_grad():
                if self.epoch == self.start_physics_at:
                    self.phy_scale = self._get_scale(
                        results["mse_loss"],
                        results["physics_loss"],
                        self.psf,
                    )
            if self.epoch >= self.start_physics_at:
                final_loss = results["mse_loss"] + self.phy_scale * results["physics_loss"]
            else:
                final_loss = results["mse_loss"]

            results["loss"] = final_loss
            return results

    def valid_step(self, batch):
        """
        This method overrides :func:`Trainer.valid_step <molearn.trainers.Trainer.valid_step>` and adds an additional 'Physics_loss' term.

        Differently to :func:`train_step <molearn.trainers.OpenMM_Physics_Trainer.train_step>` this method sums the logs of mse_loss and physics_loss ``final_loss = torch.log(results['mse_loss'])+scale*torch.log(results['physics_loss'])``

        Called from super class :func:`Trainer.valid_epoch<molearn.trainer.Trainer.valid_epoch>` on every mini-batch.

        :param torch.Tensor batch: Tensor of shape [Batch size, 3, Number of Atoms]. A mini-batch of protein frames normalised. To recover original data multiple by ``self.std``.
        :returns:  Return loss. The dictionary must contain an entry with key ``'loss'`` that will be the score via which the best checkpoint is determined.
        :rtype: dict

        """
        if self.for_transformer:
            true_coords = batch
            predicted_coords = self.autoencoder(true_coords)
            
            rec_loss = F.mse_loss(predicted_coords, true_coords)
            dist_loss = compute_distance_loss(predicted_coords, true_coords)
            torsion_loss = compute_torsion_loss(predicted_coords, true_coords)
            
            results = {
                "rec_loss": rec_loss,
                "dist_loss": dist_loss,
                "torsion_loss": torsion_loss,
            }

            if self.epoch >= self.start_physics_at:
                phys_loss = self._compute_transformer_physics_loss(predicted_coords)
                results["phys_loss"] = phys_loss
            else:
                results["phys_loss"] = torch.tensor(0.0, device=self.device)
            
            # Use log-sum for a stable validation metric.
            # Add a small epsilon to prevent log(0).
            w = self.loss_weights
            log_loss = (
                (w['w_rec'] * torch.log(results['rec_loss'] + 1e-8)) +
                (w['w_dist'] * torch.log(results['dist_loss'] + 1e-8)) +
                (w['w_torsion'] * torch.log(results['torsion_loss'] + 1e-8))
            )
            # Only add physics loss to the log sum if it's active
            if self.epoch >= self.start_physics_at:
                log_loss += (w['w_phys'] * torch.log(results['phys_loss'] + 1e-8))

            results["loss"] = log_loss
            return results

        results = self.common_step(batch)
        results.update(self.common_physics_step(batch, self._internal["encoded"]))
        # scale = (self.psf*results['mse_loss'])/(results['physics_loss'] +1e-5)
        final_loss = torch.log(results["mse_loss"]) + self.psf * torch.log(
            results["physics_loss"]
        )
        results["loss"] = final_loss
        return results


if __name__ == "__main__":
    pass
