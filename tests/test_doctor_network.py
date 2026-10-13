"""Offline node-network checks with fake sysfs and a local remote-command stub."""
import tempfile
import unittest
from pathlib import Path
from test_launcher_build import shell


class HcaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for hca, state in [('good', '4: ACTIVE'), ('down', '1: DOWN')]:
            path = self.root / 'sys/class/infiniband' / hca / 'ports/1/state'
            path.parent.mkdir(parents=True)
            path.write_text(state + '\n')

    def run_check(self, code, **settings):
        return shell('remote_on() { shift; bash -c "$*"; }\n' + code, self.root, {'DOCTOR_SYSFS_ROOT': str(self.root / 'sys'),
                                      'IB_HCA': 'good', **settings})

    def test_active_passes(self):
        out = self.run_check('doctor_network_checks')
        self.assertIn('head: ACTIVE HCAs: good', out)

    def test_down_missing_and_b12x_warn(self):
        out = self.run_check('doctor_network_checks', IB_HCA='down,missing',
                             EXTRA_CONTAINER_ENV='OTHER=1 B12X_ROCE_HCA=good,absent')
        for hca in ('down', 'missing', 'absent'):
            self.assertIn(hca, out)
        self.assertIn('ACTIVE HCAs: good', out)

    def test_strict_fails(self):
        out = self.run_check('if doctor_network_checks; then echo PASSED; else echo FAILED; fi',
                             DOCTOR_STRICT='1', IB_HCA='down')
        self.assertIn('FAILED', out)

    def test_every_worker_uses_its_own_sysfs(self):
        other = self.root / 'worker-sys'
        other.mkdir()
        out = self.run_check('''
          WORKER_HOSTS=(worker1 worker2 worker3)
          remote_on() { local host="$1"; shift; echo "VISIT $host"; bash -c "$*"; }
          doctor_network_checks
        ''')
        for node in ('head', 'worker1', 'worker2', 'worker3'):
            self.assertIn(node + ': ACTIVE HCAs: good', out)
        self.assertEqual(out.count('VISIT'), 3)

class InterfaceTests(HcaTests):
    def test_socket_interfaces_present(self):
        (self.root / 'sys/class/net/eth0').mkdir(parents=True)
        out = self.run_check('doctor_network_checks', GLOO_SOCKET_IFNAME='eth0',
                             NCCL_SOCKET_IFNAME='=eth0', DOCTOR_STRICT='1')
        self.assertIn('head: socket interfaces OK', out)

    def test_missing_socket_warns(self):
        out = self.run_check('doctor_network_checks', GLOO_SOCKET_IFNAME='missing-gloo',
                             NCCL_SOCKET_IFNAME='missing-nccl')
        for node in ('head', '10.0.0.2', '10.0.0.3'):
            for name in ('missing-gloo', 'missing-nccl'):
                self.assertIn(node + ': socket interface ' + name + ' missing', out)

    def test_missing_socket_strict_fails(self):
        out = self.run_check('if doctor_network_checks; then echo PASSED; else echo FAILED; fi',
                             IB_HCA='good', DOCTOR_STRICT='1', GLOO_SOCKET_IFNAME='absent')
        self.assertIn('FAILED', out)

if __name__ == '__main__':
    unittest.main()
