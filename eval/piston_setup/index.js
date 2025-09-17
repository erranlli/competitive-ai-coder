#!/usr/bin/env node
require('nocamel');
const Logger = require('logplease');
const express = require('express');
const expressWs = require('express-ws');
const globals = require('./globals');
const config = require('./config');
const path = require('path');
const fs = require('fs/promises');
const fss = require('fs');
const body_parser = require('body-parser');
const runtime = require('./runtime');

const logger = Logger.create('index');
const app = express();
expressWs(app);

(async () => {
    logger.info('Setting loglevel to', config.log_level);
    Logger.setLogLevel(config.log_level);
    logger.debug('Ensuring data directories exist');

    Object.values(globals.data_directories).for_each(dir => {
        let data_path = path.join(config.data_directory, dir);

        logger.debug(`Ensuring ${data_path} exists`);

        if (!fss.exists_sync(data_path)) {
            logger.info(`${data_path} does not exist.. Creating..`);

            try {
                fss.mkdir_sync(data_path);
            } catch (e) {
                logger.error(`Failed to create ${data_path}: `, e.message);
            }
        }
    });

    logger.info('Loading packages');
    const pkgdir = path.join(
        config.data_directory,
        globals.data_directories.packages
    );

    const pkglist = await fs.readdir(pkgdir);

    const languages = await Promise.all(
        pkglist.map(lang => {
            return fs.readdir(path.join(pkgdir, lang)).then(x => {
                return x.map(y => path.join(pkgdir, lang, y));
            });
        })
    );

    const installed_languages = languages
        .flat()
        .filter(pkg =>
            fss.exists_sync(path.join(pkg, globals.pkg_installed_file))
        );

    installed_languages.for_each(pkg => runtime.load_package(pkg));

    logger.info('Starting API Server');
    logger.debug('Constructing Express App');
    logger.debug('Registering middleware');

    // Increase body-parser limits to allow large checkers / testcases.
    // Adjust these values if you need larger or smaller limits.
    const JSON_LIMIT = process.env.PISTON_JSON_LIMIT || '512mb';
    const URLENCODE_LIMIT = process.env.PISTON_URLENCODE_LIMIT || '512mb';

    console.log(`[INFO] Setting JSON body limit to: ${JSON_LIMIT}`);
    console.log(`[INFO] Setting URL Encoded body limit to: ${URLENCODE_LIMIT}`);

    app.use(body_parser.urlencoded({ extended: true, limit: URLENCODE_LIMIT }));
    app.use(body_parser.json({ limit: JSON_LIMIT }));

    // Improved error handler: map payload-too-large to 413 with clearer message
    app.use((err, req, res, next) => {
        if (!err) {
            return next();
        }

        // Log error for debugging
        logger.warn('Express middleware error:', err && err.message ? err.message : String(err));

        // raw-body / body-parser sets err.type === 'entity.too.large' in many versions
        if (err.type === 'entity.too.large' || (err.status === 413) ||
            (err.message && typeof err.message === 'string' && err.message.toLowerCase().includes('request entity too large')) ||
            (err.stack && err.stack.includes('PayloadTooLargeError'))) {
            return res.status(413).send({
                error: 'PayloadTooLargeError',
                message: 'Request body is too large. Consider increasing server body-parser limits or sending smaller payloads.',
                detail: err.message,
                stack: err.stack,
            });
        }

        // Other parse errors: return 400 with the stack to aid debugging (this is what the original did)
        return res.status(400).send({
            stack: err.stack,
            message: err.message,
        });
    });

    app.use((err, req, res, next) => {
        return res.status(400).send({
            stack: err.stack,
        });
    });

    logger.debug('Registering Routes');

    const api_v2 = require('./api/v2');
    app.use('/api/v2', api_v2);

    const { version } = require('../package.json');

    app.get('/', (req, res, next) => {
        return res.status(200).send({ message: `Piston v${version}` });
    });

    app.use((req, res, next) => {
        return res.status(404).send({ message: 'Not Found' });
    });

    logger.debug('Calling app.listen');
    const [address, port] = config.bind_address.split(':');

    const server = app.listen(port, address, () => {
        logger.info('API server started on', config.bind_address);
    });

    process.on('SIGTERM', () => {
        server.close();
        process.exit(0)
    });
})();
