class Media {
    constructor(unprocessedMediaPath, date, latitudeCoords, longitudeCoords, confidence = 0.0, count = 0) {
        this._unprocessedMediaPath = unprocessedMediaPath;
        this._processedMediaPath = '';
        this._date = date;
        this._latitudeCoords = latitudeCoords;
        this._longitudeCoords = longitudeCoords;
        this._flagged = false;
        this._confidence = Number(confidence);
        this._count = Number.isFinite(Number(count)) ? Math.trunc(Number(count)) : 0;
    }

    get latitude() {
        return this._latitudeCoords;
    }

    get longitude() {
        return this._longitudeCoords;
    }

    get date() {
        return this._date;
    }

    get unprocessed() {
        return this._unprocessedMediaPath;
    }

    set processed(processedMediaPath) {
        this._processedMediaPath = processedMediaPath;
    }

    get processed() {
        return this._processedMediaPath;
    }

    get flagged() {
        return this._flagged;
    }

    set flagged(value) {
        this._flagged = value;
    }

    get confidence() {
        return this._confidence;
    }

    set confidence(value) {
        this._confidence = Number(value);
    }

    get count() {
        return this._count;
    }

    set count(value) {
        const parsedValue = Number(value);
        this._count = Number.isFinite(parsedValue) ? Math.trunc(parsedValue) : 0;
    }

    toJSON() {
        return {
            unprocessedMediaPath: this._unprocessedMediaPath,
            processedMediaPath: this._processedMediaPath,
            date: this._date,
            latitudeCoords: this._latitudeCoords,
            longitudeCoords: this._longitudeCoords,
            flagged: this._flagged,
            confidence: this._confidence,
            count: this._count
        };
    }
}

if (typeof window !== 'undefined') {
    window.Media = Media;
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = Media;
}
