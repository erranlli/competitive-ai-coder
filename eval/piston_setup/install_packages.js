// install_packages.js
const { install, list } = require('./src/packages.js');

async function main() {
    console.log('--- Starting pre-installation of packages ---');
    
    console.log('Installing latest Python...');
    await install('python');
    
    console.log('Installing Python 3.9.4...');
    await install('python', '3.9.4');
    
    console.log('--- Package installation complete ---');
    const installed = await list();
    console.log('Installed packages:', installed);
}

main().catch(err => {
    console.error('Package installation failed:', err);
    process.exit(1);
});
