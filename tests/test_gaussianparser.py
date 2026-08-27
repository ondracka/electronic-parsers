#
# Copyright The NOMAD Authors.
#
# This file is part of NOMAD. See https://nomad-lab.eu for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pytest
import numpy as np
from pathlib import Path

from nomad.datamodel import EntryArchive
from electronicparsers.gaussian import GaussianParser
from electronicparsers.gaussian.parser import _get_vdw_method


def approx(value, abs=0, rel=1e-6):
    return pytest.approx(value, abs=abs, rel=rel)


@pytest.fixture(scope='module')
def parser():
    return GaussianParser()


def test_scf_spinpol(parser):
    archive = EntryArchive()
    parser.parse('tests/data/gaussian/Al_scf/Al.out', archive, None)

    sec_runs = archive.run
    assert len(sec_runs) == 1
    assert sec_runs[0].clean_end
    assert sec_runs[0].x_gaussian_program_implementation == 'EM64L-G09RevB.01'
    assert sec_runs[0].x_gaussian_number_of_processors == '8'
    assert (
        len(
            sec_runs[0]
            .x_gaussian_section_orbital_symmetries[0]
            .x_gaussian_alpha_symmetries
        )
        == 50
    )

    sec_methods = sec_runs[0].method
    assert (
        sec_methods[0].electrons_representation[0].basis_set[0].atom_centered[0].name
        == 'AUG-CC-PVTZ'
    )
    assert len(sec_methods) == 1
    assert sec_methods[0].dft.xc_functional.hybrid[0].name == 'HYB_GGA_XC_B3LYP'
    assert sec_methods[0].electronic.charge.magnitude == -1
    assert sec_methods[0].electronic.van_der_waals_method == ''
    assert sec_methods[0].electronic.m_to_dict()['van_der_waals_method'] == ''

    sec_systems = sec_runs[0].system
    assert len(sec_systems) == 1
    assert sec_systems[0].atoms.periodic == [False, False, False]
    assert sec_systems[0].atoms.labels == ['Al']

    sec_sccs = sec_runs[0].calculation
    assert len(sec_sccs) == 1
    assert sec_sccs[0].energy.total.value.magnitude == approx(-1.05675722e-15)
    assert len(sec_sccs[0].x_gaussian_section_hybrid_coeffs) == 1
    assert np.shape(sec_sccs[0].eigenvalues[0].occupations[0][0]) == (50,)
    assert np.shape(sec_sccs[0].eigenvalues[0].energies[0][0]) == (50,)
    assert sec_sccs[0].eigenvalues[0].occupations[0][0][7] == 0
    assert sec_sccs[0].eigenvalues[0].energies[0][0][-5].magnitude == approx(
        4.64011991e-18
    )
    assert sec_sccs[0].x_gaussian_section_molecular_multipoles[
        0
    ].x_gaussian_molecular_multipole_values[4] == approx(-9.36527896e-39)
    assert len(sec_sccs[0].scf_iteration) == 1
    assert sec_sccs[0].scf_iteration[0].energy.total.value.magnitude == approx(
        -1.05675722e-15
    )
    assert sec_sccs[0].calculation_converged


def test_error_termination_sets_clean_end_false(parser, tmp_path):
    source = Path('tests/data/gaussian/Al_scf/Al.out').read_text()
    normal_line = next(
        line for line in source.splitlines(keepends=True) if 'Normal termination' in line
    )
    output = tmp_path / 'Al_error.out'
    output.write_text(
        source.replace(
            normal_line,
            ' Error termination via Lnk1e in /gaussian/l9999.exe.\n',
        )
    )

    archive = EntryArchive()
    parser.parse(str(output), archive, None)
    assert len(archive.run) == 1
    assert archive.run[0].clean_end is False


@pytest.mark.parametrize(
    'keyword, expected',
    [
        ('GD2', 'DFT-D2'),
        ('GD3', 'DFT-D3(0)'),
        ('GD3BJ', 'DFT-D3(BJ)'),
        ('PFD', 'PFD'),
        ('None', ''),
    ],
)
def test_empirical_dispersion_mapping(keyword, expected):
    settings = f'# PBEPBE/6-31G EmpiricalDispersion={keyword}'
    assert _get_vdw_method(settings) == expected


def test_empirical_dispersion_route(parser, tmp_path):
    source = Path('tests/data/gaussian/Al_scf/Al.out').read_text()
    source = source.replace(
        '#B3LYP/aug-cc-pVTZ scf=verytight integral(grid=ultrafine)',
        '#B3LYP/aug-cc-pVTZ EmpiricalDispersion=GD3BJ scf=verytight '
        'integral(grid=ultrafine)',
        1,
    )
    output = tmp_path / 'Al.out'
    output.write_text(source)

    archive = EntryArchive()
    parser.parse(str(output), archive, None)
    electronic = archive.run[0].method[0].electronic
    assert electronic.van_der_waals_method == 'DFT-D3(BJ)'
    assert electronic.m_to_dict()['van_der_waals_method'] == 'DFT-D3(BJ)'


def test_scf_multirun(parser):
    archive = EntryArchive()
    parser.parse('tests/data/gaussian/Al_multistep/m61b5.out', archive, None)

    sec_runs = archive.run
    assert len(sec_runs) == 2
    assert len(sec_runs[0].calculation) == 6
    assert len(sec_runs[1].calculation) == 1
    assert len(sec_runs[0].calculation[2].scf_iteration) == 11
    assert len(sec_runs[1].calculation[0].scf_iteration) == 1
    assert len(sec_runs[0].system) == 6
    assert len(sec_runs[1].system) == 1
    assert len(sec_runs[0].method) == 1
    assert len(sec_runs[1].method) == 1

    sec_scc = sec_runs[0].calculation[4]
    assert sec_scc.forces.total.value_raw[0][2].magnitude == approx(-9.69697756e-14)
    assert sec_scc.scf_iteration[3].energy.change.magnitude == approx(-8.82412332e-27)

    sec_thermochem = sec_runs[1].x_gaussian_section_thermochem[0]
    sec_thermo = sec_runs[1].calculation[0].thermodynamics[0]
    assert sec_thermo.temperature.magnitude == approx(298.15)
    assert sec_thermochem.x_gaussian_moments[1] == approx(8.59409221e-45)
    assert sec_thermochem.x_gaussian_thermal_correction_free_energy == approx(
        -1.00274129e-19
    )


def test_mp(parser):
    archive = EntryArchive()
    parser.parse('tests/data/gaussian/NO_mp/onno.out', archive, None)

    sec_sccs = archive.run[0].calculation
    assert len(sec_sccs) == 17
    approx(
        sec_sccs[0]
        .x_gaussian_section_moller_plesset[0]
        .x_gaussian_mp2_correction_energy.magnitude,
        -3.17820357e-18,
    )
    approx(sec_sccs[-1].energy.total.value.magnitude, -1.12849219e-15)
    approx(
        sec_sccs[3]
        .x_gaussian_section_coupled_cluster[0]
        .x_gaussian_ccsd_correction_energy.magnitude,
        -3.08257224e-18,
    )


def test_freq(parser):
    archive = EntryArchive()
    parser.parse('tests/data/gaussian/CHO_freq/prono.out', archive, None)

    sec_runs = archive.run
    assert len(sec_runs) == 2

    assert len(sec_runs[1].x_gaussian_section_frequencies) == 1
    assert np.shape(
        sec_runs[1].x_gaussian_section_frequencies[0].x_gaussian_frequencies
    ) == (33,)
    assert np.shape(
        sec_runs[1].x_gaussian_section_frequencies[0].x_gaussian_red_masses
    ) == (33,)
    assert np.shape(
        sec_runs[1].x_gaussian_section_frequencies[0].x_gaussian_normal_mode_values
    ) == (33, 13, 3)
    assert (
        sec_runs[1]
        .x_gaussian_section_frequencies[0]
        .x_gaussian_normal_mode_values[28][6][1]
        == 0.19
    )
